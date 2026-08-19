"""Roll out a trained tracking policy on many reference motions and save corrected NPZ.

Each input ``*.npz`` (MotionLoader layout) is played in a fresh environment with
``motion_file`` set to that clip. Robot states are recorded in the same layout as
``export_policy_rollouts`` / :class:`MotionLoader` expects, saved as
``<stem>_rollout.npz`` under ``--output-dir``.

Example::

  uv run python -m mjlab.scripts.export_policy_rollouts \\
    Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \\
    --checkpoint-file logs/rsl_rl/g1_tracking/run/model_500.pt \\
    --motion-directory dataset/0326/motion \\
    --output-dir dataset/0326/motion_corrected \\
    --hit-times-pkl dataset/0326/label/.../nadal_clip_hit_times.pkl
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.torch import configure_torch_backends


def _coerce_hit_frame(val: Any) -> int | None:
  """Parse a single hit-frame scalar; reject bools so ``False`` does not become 0."""
  if val is None or isinstance(val, (bool, np.bool_)):
    return None
  if isinstance(val, np.generic):
    try:
      return int(val.item())
    except (AttributeError, TypeError, ValueError):
      return None
  try:
    return int(val)
  except (TypeError, ValueError):
    return None


def _hit_frame_from_mapping(val: Any) -> int | None:
  if isinstance(val, dict):
    for key in (
      "hit_frame_relative",
      "hit_frame",
      "hit",
      "reference_hit_frame",
    ):
      if key in val:
        h = _coerce_hit_frame(val[key])
        if h is not None:
          return h
    return None
  return _coerce_hit_frame(val)


def _clip_stem_from_clip_record(rec: dict[str, Any]) -> str | None:
  for key in ("stem", "clip_stem"):
    if key in rec and rec[key] is not None:
      s = str(rec[key]).strip()
      if s:
        return Path(s).stem
  for key in (
    "clip_name",
    "reference_clip_name",
    "input",
    "reference_source",
    "generated_file",
  ):
    if key in rec and rec[key] is not None:
      s = str(rec[key]).strip()
      if s:
        return Path(s).stem
  return None


def _normalize_clip_stem(stem: str) -> str:
  """Normalize clip stem by removing prefix like clip_XXX_ to enable cross-dataset matching.

  Examples:
    - clip_001_1047_1105_r-016_p-015_y-002 -> 1047_1105_r-016_p-015_y-002
    - clip_003_12446_12506_r-024_p006_y-002 -> 12446_12506_r-024_p006_y-002
  """
  if not stem:
    return stem
  # Pattern: clip_<number>_<rest> -> keep only the rest
  import re
  match = re.match(r'^clip_\d+_(.+)$', stem)
  if match:
    return match.group(1)
  return stem


def _load_hit_times_table(path: Path | None, use_normalized_stems: bool = True) -> dict[str, int]:
  """Return mapping from clip stem (no ``.npz``) to hit frame index.

  Supports:

  * Flat ``{ "clip.npz" | stem: int | {...} }`` (legacy).
  * Nested ``{"hits": {stem: int, ...}}`` (e.g. export v2 checkpoints).
  * ``{"clips": [{"clip_name": ..., "hit_frame_relative": int}, ...]}`` (legged_gym style).

  When ``use_normalized_stems=True`` (default), stems are normalized to ignore
  dataset-specific prefixes like ``clip_XXX_`` for cross-dataset matching.
  """
  if path is None or not path.is_file():
    return {}
  with path.open("rb") as f:
    raw = pickle.load(f)
  out: dict[str, int] = {}

  def put_stem(stem: str, val: Any) -> None:
    if not stem:
      return
    h = _hit_frame_from_mapping(val)
    if h is not None:
      # Store under both original and normalized stems for flexibility
      out[stem] = h
      if use_normalized_stems:
        norm = _normalize_clip_stem(stem)
        if norm and norm != stem:
          out[norm] = h

  if isinstance(raw, dict):
    hits_sub = raw.get("hits")
    if isinstance(hits_sub, dict):
      for k, v in hits_sub.items():
        put_stem(Path(str(k)).stem, v)

    clips = raw.get("clips")
    if isinstance(clips, list):
      for rec in clips:
        if isinstance(rec, dict):
          st = _clip_stem_from_clip_record(rec)
          if st is not None:
            put_stem(st, rec)

    skip_top = {
      "hits",
      "clips",
      "version",
      "metadata",
      "last_clip_index",
      "last_stem",
      "task_id",
      "checkpoint",
      "clips_so_far",
    }
    for k, v in raw.items():
      if k in skip_top:
        continue
      put_stem(Path(str(k)).stem, v)
  elif isinstance(raw, list):
    for rec in raw:
      if isinstance(rec, dict):
        st = _clip_stem_from_clip_record(rec)
        if st is not None:
          put_stem(st, rec)

  return out


def _snapshot_robot(env: RslRlVecEnvWrapper, motion_cmd: MotionCommand) -> dict[str, np.ndarray]:
  """One timestep of robot state in MotionLoader NPZ layout (env-local body positions)."""
  uw = env.unwrapped
  robot = uw.scene["robot"]
  origins = uw.scene.env_origins[0:1, :].to(uw.device)
  idx = motion_cmd.body_indexes
  joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
  joint_vel = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
  body_pos_w = (robot.data.body_link_pos_w[0, idx] - origins).detach().cpu().numpy().astype(
    np.float32
  )
  body_quat_w = robot.data.body_link_quat_w[0, idx].detach().cpu().numpy().astype(np.float32)
  body_lin_vel_w = robot.data.body_link_lin_vel_w[0, idx].detach().cpu().numpy().astype(
    np.float32
  )
  body_ang_vel_w = robot.data.body_link_ang_vel_w[0, idx].detach().cpu().numpy().astype(
    np.float32
  )
  return {
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
  }


def _collect_motion_npz_paths(motion_dir: Path, pattern: str) -> list[Path]:
  paths = sorted(motion_dir.glob(pattern))
  return [p for p in paths if p.is_file() and p.suffix == ".npz" and "_rollout" not in p.stem]


def _flatten_aligned_bodies(
  ref_b: np.ndarray, roll_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """``ref_b`` (N_r, 3), ``roll_b`` (T, N_t, 3) -> same leading bodies, flattened."""
  if ref_b.ndim != 2 or roll_b.ndim != 3:
    raise ValueError(f"Expected ref (N,3) and roll (T,N,3), got {ref_b.shape} {roll_b.shape}")
  n = int(min(ref_b.shape[0], roll_b.shape[1]))
  if n <= 0:
    raise ValueError("Empty body layout for comparison.")
  ref_f = ref_b[:n].astype(np.float32).reshape(-1)
  roll_f = roll_b[:, :n].astype(np.float32).reshape(roll_b.shape[0], -1)
  return ref_f, roll_f


def _flatten_aligned_joints(ref_j: np.ndarray, roll_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """``ref_j`` (D_r,), ``roll_j`` (T, D_t) -> truncate to common dof."""
  if ref_j.ndim != 1 or roll_j.ndim != 2:
    raise ValueError(f"Expected ref (D,) and roll (T,D), got {ref_j.shape} {roll_j.shape}")
  d = int(min(ref_j.shape[0], roll_j.shape[1]))
  if d <= 0:
    raise ValueError("Empty joint layout for comparison.")
  return ref_j[:d].astype(np.float32), roll_j[:, :d].astype(np.float32)


def rollout_hit_frame_time_window(
  rollout: dict[str, np.ndarray],
  ref_npz: Path,
  ref_hit_frame: int,
  *,
  step_dt: float,
  extra_steps: int,
) -> int:
  """Pick rollout frame near reference *motion time* at hit, then refine by pose.

  Discrete motion time at reference hit is ``t_hit ≈ ref_hit_frame * step_dt``.
  Candidate rollout indices are ``i`` with ``|i - ref_hit_frame| <= extra_steps``
  (same clock: one env step per reference row). Among those, return ``i`` whose
  pose (aligned bodies, else joints) is closest to the reference pose at
  ``ref_hit_frame``.

  This avoids (1) comparing different body counts between ref NPZ and rollout, and
  (2) searching the whole clip when only the impact neighbourhood matters.
  """
  if step_dt <= 0:
    raise ValueError(f"step_dt must be positive, got {step_dt}")
  t_roll = int(rollout["joint_pos"].shape[0])
  h = int(np.clip(ref_hit_frame, 0, t_roll - 1))
  k = max(0, int(extra_steps))
  lo = max(0, h - k)
  hi = min(t_roll - 1, h + k)
  idx = np.arange(lo, hi + 1, dtype=np.int64)

  with np.load(str(ref_npz), mmap_mode="r") as ref:
    t_ref = int(ref["joint_pos"].shape[0])
    h_ref = int(np.clip(ref_hit_frame, 0, t_ref - 1))
    if t_roll < t_ref:
      h_ref = min(h_ref, t_roll - 1)
    if "body_pos_w" in ref.files and "body_pos_w" in rollout:
      ref_bh = np.asarray(ref["body_pos_w"][h_ref], dtype=np.float32)
      roll_b = rollout["body_pos_w"].astype(np.float32)
      target, r_full = _flatten_aligned_bodies(ref_bh, roll_b)
    else:
      ref_jh = np.asarray(ref["joint_pos"][h_ref], dtype=np.float32).reshape(-1)
      roll_j = rollout["joint_pos"].astype(np.float32)
      target, r_full = _flatten_aligned_joints(ref_jh, roll_j)
  r = r_full[idx]
  dist = np.linalg.norm(r - target[None, :], axis=1)
  j = int(np.argmin(dist))
  return int(idx[j])


def _atomic_pickle(path: Path, obj: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("wb") as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
  os.replace(tmp, path)


@dataclass(frozen=True)
class ExportPolicyRolloutsConfig:
  """Export policy rollouts for every motion NPZ in a directory."""

  checkpoint_file: Path
  """Local policy checkpoint (``*.pt``)."""

  motion_directory: Path
  """Directory containing reference ``*.npz`` clips (not ``*_rollout.npz``)."""

  output_dir: Path
  """Directory to write ``<stem>_rollout.npz``."""

  hit_times_pkl: Path | None = None
  """Optional pickle of per-clip hit frames (keys: clip path or stem)."""

  hit_times_output_pkl: Path | None = None
  """Where to save generated hit times (defaults to ``output_dir/generated_hit_times.pkl``)."""

  hit_time_scaling_factor: float = 1.0
  """Scale factor applied to reference hit frame from ``hit_times_pkl``.

  Effective frame used for rollout alignment is:
  ``scaled_hit_frame = round(raw_hit_frame * (50 / 30) / hit_time_scaling_factor)``.
  The hit labels are authored at 30fps while simulation runs at 50fps, so the
  base conversion is ``* (50/30) = * (5/3)``. ``hit_time_scaling_factor`` is
  then applied as an additional adjustment. For example, ``0.75`` means wait
  longer (divide by 0.75 after fps conversion).
  """

  motion_glob: str = "*.npz"
  """Glob under ``motion_directory``."""

  device: str | None = None
  """Torch device; default CUDA if available."""

  skip_existing: bool = True
  """If true, skip clips whose ``<stem>_rollout.npz`` already exists in ``output_dir``."""

  hit_match_extra_steps: int = 1
  """Around reference ``hit_frame`` (same discrete clock as ``step_dt``), search rollout
  frames ``hit ± hit_match_extra_steps`` and pick the one whose pose is closest to
  the reference at hit. Use ``2`` or ``3`` if tracking drifts in time."""

  write_summary_json: bool = True
  """Write ``export_summary.json`` under ``output_dir``."""


def run_export(task_id: str, cfg: ExportPolicyRolloutsConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  motion_dir = cfg.motion_directory.expanduser().resolve()
  out_dir = cfg.output_dir.expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  ckpt = cfg.checkpoint_file.expanduser().resolve()
  if not ckpt.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

  hit_table_path = cfg.hit_times_pkl.expanduser().resolve() if cfg.hit_times_pkl else None
  if hit_table_path is not None and not hit_table_path.is_file():
    raise FileNotFoundError(f"Hit-times table not found: {hit_table_path}")
  hit_ref = _load_hit_times_table(
    hit_table_path,
    use_normalized_stems=True,
  )
  # Debug: show hit_ref stats
  print(f"[DEBUG] hit_ref entries: {len(hit_ref)}")
  if hit_ref:
    sample_keys = list(hit_ref.keys())[:5]
    print(f"[DEBUG] sample keys: {sample_keys}")
  if cfg.hit_time_scaling_factor <= 0:
    raise ValueError(
      f"hit_time_scaling_factor must be > 0, got {cfg.hit_time_scaling_factor}"
    )
  hit_sf = float(cfg.hit_time_scaling_factor)
  fps_scale = 50.0 / 30.0
  if hit_table_path is not None:
    print(
      f"[Export] Hit table {hit_table_path.name}: parsed {len(hit_ref)} stem(s) "
      f"(empty usually means pkl layout not recognised or stem mismatch)."
    )
  hit_out_path = (
    cfg.hit_times_output_pkl.expanduser().resolve()
    if cfg.hit_times_output_pkl is not None
    else out_dir / "generated_hit_times.pkl"
  )

  paths = _collect_motion_npz_paths(motion_dir, cfg.motion_glob)
  if not paths:
    raise FileNotFoundError(f"No motion npz matching {cfg.motion_glob!r} under {motion_dir}")

  env_cfg = load_env_cfg(task_id, play=True)
  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise ValueError(f"Task {task_id!r} has no MotionCommandCfg at commands['motion'].")

  env_cfg.terminations = {}
  env_cfg.scene.num_envs = 1
  motion_cfg.sampling_mode = "start"
  motion_cfg.motion_file = str(paths[0].resolve())
  motion_cfg.motion_directory = ""
  motion_cfg.motion_files = ()

  agent_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner

  generated_hits: dict[str, int] = {}
  summary: dict[str, Any] = {"task_id": task_id, "checkpoint": str(ckpt), "clips": []}

  print(f"[Export] Checkpoint: {ckpt.name}  |  clips: {len(paths)}  |  device: {device}")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)
  motion_cmd = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  loaded_motion = paths[0].resolve()

  def _save_hit_progress(clip_index: int, latest_stem: str) -> None:
    payload: dict[str, Any] = {
      "version": 2,
      "task_id": task_id,
      "checkpoint": str(ckpt),
      "hits": dict(generated_hits),
      "clips": list(summary["clips"]),
      "last_clip_index": clip_index,
      "last_stem": latest_stem,
    }
    _atomic_pickle(hit_out_path, payload)
    snap = out_dir / f"generated_hit_{clip_index:04d}.pkl"
    _atomic_pickle(
      snap,
      {
        "clip_index": clip_index,
        "stem": latest_stem,
        "hits_so_far": dict(generated_hits),
        "clips_so_far": len(summary["clips"]),
      },
    )

  pbar = tqdm(
    enumerate(paths, start=1),
    total=len(paths),
    desc="Export rollouts",
    unit="clip",
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
  )
  try:
    for clip_i, motion_path in pbar:
      stem = motion_path.stem
      out_path = out_dir / f"{stem}_rollout.npz"
      pbar.set_postfix_str(stem[:32] + ("…" if len(stem) > 32 else ""))

      if cfg.skip_existing and out_path.is_file():
        tqdm.write(f"[Export] skip (exists): {out_path.name}")
        norm_stem = _normalize_clip_stem(stem)
        hit_frame_lookup = hit_ref.get(stem) or hit_ref.get(norm_stem)
        summary["clips"].append(
          {
            "input": str(motion_path),
            "output": str(out_path),
            "T": None,
            "hit_frame": hit_frame_lookup,
            "skipped": True,
          }
        )
        continue

      ref_npz_name = motion_path.name
      t_mmap = int(np.load(motion_path, mmap_mode="r")["joint_pos"].shape[0])

      target = motion_path.resolve()
      if target != loaded_motion:
        motion_cmd.swap_reference_motion_file(str(target))
        loaded_motion = target
      _, _ = env.reset()
      t_cmd = int(motion_cmd.motion.time_step_total)

      assert t_cmd == t_mmap, (
        f"{motion_path}: motion command T={t_cmd} vs file T={t_mmap}"
      )

      obs = env.get_observations()
      env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)

      frames: list[dict[str, np.ndarray]] = []
      frames.append(_snapshot_robot(env, motion_cmd))

      for _ in range(t_mmap - 1):
        with torch.inference_mode():
          actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        if dones.any():
          tqdm.write(
            f"[Export][WARN] early done in {motion_path.name} at step {len(frames)}"
          )
          break
        frames.append(_snapshot_robot(env, motion_cmd))

      stack = {k: np.stack([f[k] for f in frames], axis=0) for k in frames[0].keys()}
      save_kw: dict[str, Any] = dict(stack)
      save_kw["reference_clip_name"] = np.array(ref_npz_name, dtype=object)

      norm_stem = _normalize_clip_stem(stem)
      raw_hit_frame = hit_ref.get(stem) or hit_ref.get(norm_stem)
      hit_frame = (
        int(round(float(raw_hit_frame) * fps_scale / hit_sf))
        if raw_hit_frame is not None
        else None
      )
      if hit_frame is not None and hit_frame < len(frames):
        rel = int(hit_frame)
        save_kw["reference_hit_frame"] = np.array([rel], dtype=np.int64)

      step_dt = float(env.unwrapped.step_dt)
      rollout_hit_frame: int | None = None
      if hit_frame is not None:
        hf = int(hit_frame)
        if 0 <= hf < t_mmap and stack["joint_pos"].shape[0] > 0:
          try:
            rollout_hit_frame = rollout_hit_frame_time_window(
              stack,
              motion_path,
              hf,
              step_dt=step_dt,
              extra_steps=cfg.hit_match_extra_steps,
            )
            t_hit = hf * step_dt
            half = cfg.hit_match_extra_steps * step_dt
            ratio = rollout_hit_frame / float(hf) if hf > 0 else float("nan")
            save_kw["rollout_hit_frame"] = np.array([rollout_hit_frame], dtype=np.int64)
            save_kw["hit_frame_relative"] = np.array([rollout_hit_frame], dtype=np.int64)
            generated_hits[stem] = rollout_hit_frame
            tqdm.write(
              f"[Export] {stem}: ref_hit_frame={hf} (raw={raw_hit_frame}, fps_scale={fps_scale:.4f}, "
              f"sf={hit_sf:.4f}, t≈{t_hit:.4f}s), "
              f"rollout_hit_frame={rollout_hit_frame} "
              f"(search |i-{hf}|≤{cfg.hit_match_extra_steps}, "
              f"Δt≤±{half:.4f}s), ratio={ratio:.3f}"
            )
          except Exception as exc:
            tqdm.write(
              f"[Export][WARN] {stem}: rollout hit frame failed: {exc}"
            )
        else:
          tqdm.write(
            f"[Export] {stem}: Reference hit frame = {hf} (raw={raw_hit_frame}, fps_scale={fps_scale:.4f}, "
            f"sf={hit_sf:.4f}) "
            f"(out of range for T={t_mmap} or empty rollout, skip compare)"
          )

      np.savez_compressed(out_path, **save_kw)

      clip_summary: dict[str, Any] = {
        "input": str(motion_path),
        "output": str(out_path),
        "T": len(frames),
        "hit_frame": hit_frame,
        "raw_hit_frame": raw_hit_frame,
        "fps_scale_30_to_50": fps_scale,
        "hit_time_scaling_factor": hit_sf,
        "skipped": False,
      }
      if rollout_hit_frame is not None:
        clip_summary["rollout_hit_frame"] = rollout_hit_frame
      summary["clips"].append(clip_summary)
      _save_hit_progress(clip_i, stem)
      tqdm.write(
        f"[Export] [{clip_i}/{len(paths)}] saved {out_path.name}  (T={len(frames)}, hits={len(generated_hits)})"
      )
  finally:
    env.close()

  tqdm.write(
    f"[Export] Hit-time progress -> {hit_out_path} ({len(generated_hits)} hit entries, {len(paths)} clips)"
  )

  if cfg.write_summary_json:
    summary_path = out_dir / "export_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
    tqdm.write(f"[Export] Summary JSON -> {summary_path}")


def main() -> None:
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportPolicyRolloutsConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
