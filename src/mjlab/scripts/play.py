"""Script to play RL agent with RSL-RL."""

import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Literal

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.adapt_tennis.mdp.events import apply_motion_reference_warmup
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


def _resolve_play_motion_file(cfg: "PlayConfig") -> Path | None:
  """Return a single ``.npz`` path only when ``--motion-file`` is set (not a directory)."""
  if cfg.motion_file is None:
    return None
  p = Path(cfg.motion_file).expanduser()
  if not p.is_file():
    return None
  return p.resolve()


def _validate_play_motion_directory(cfg: "PlayConfig") -> tuple[Path, int] | None:
  """Return ``(resolved_dir, num_npz)`` if ``--motion-directory`` is set (multi-clip play)."""
  if cfg.motion_directory is None:
    return None
  d = Path(cfg.motion_directory).expanduser().resolve()
  if not d.is_dir():
    raise FileNotFoundError(f"motion_directory is not a directory: {d}")
  nzs = sorted(x for x in d.glob("*.npz") if x.is_file())
  if not nzs:
    raise FileNotFoundError(f"No .npz files under {d}")
  return d, len(nzs)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  onnx_file: str | None = None
  """Optional ONNX policy path. If set, inference uses ONNXRuntime (no .pt runner load)."""
  motion_file: str | None = None
  motion_directory: str | None = None
  """Directory of ``*.npz``: loads **all** clips in ``MotionCommand`` (switches on resample)."""
  init_state_directory: str | None = None
  """Directory of single-frame init ``*.npz`` for ``InitStateCommand`` (grasp-only play)."""

  motion_clip_index: int = 0
  """With ``--motion-file`` only: if path is a **directory**, pick this sorted ``*.npz`` index (rare)."""
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  video_duration_s: float | None = None
  """If set with ``--video``, overrides ``video_length`` as ``ceil(duration_s / step_dt)``."""
  video_folder: str | None = None
  """Directory for recorded ``.mp4``. Default: ``<checkpoint_dir>/videos/play``."""
  video_headless: bool = False
  """If True, record offscreen then exit (no native/viser window). Implies ``--video``."""
  debug_vis: bool = True
  """Draw ghost motion / court / collision overlays. Set ``False`` for clean videos."""
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  viser_port: int | None = None
  """Optional Viser server port. Only applies when using viewer='viser'."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  explore_jit: bool = False
  """If True, export `modeljit_xxx.pt` from loaded checkpoint during play."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False
  collect_dof_pos_n: int = 0
  """If > 0, run headless collection mode for N rollouts and save N ``.npz`` files."""
  collect_per_motion_directory: bool = False
  """With ``--motion-directory``: one replay ``.npz`` per input ``*.npz`` (full clip by default)."""
  collect_motion_time_start: float = 0.0
  """Collect window start (seconds) on motion timeline."""
  collect_motion_time_end: float = 1.0
  """Collect window end (seconds) on motion timeline."""
  collect_sample_dt: float = 0.02
  """Dense sampling period (seconds) on motion timeline during rollout (default 50 Hz)."""
  collect_output_hz: float = 0.0
  """Optional resample Hz before save (``collect_dof_pos_n`` mode). ``<= 0`` keeps dense rate."""
  collect_output_dir: str = "logs/play_dof_pos"
  """Output directory for collected trajectories."""
  collect_max_steps_per_rollout: int = 20000
  """Safety cap for one collection rollout."""

  motion_warmup_steps: int | None = None
  """Hold motion reference at frame 0 for this many steps (``0`` or unset = off)."""
  motion_warmup_step_source: Literal["env", "learning_iteration", "episode"] | None = None
  """Warmup counter: use ``episode`` for play (resets each env episode); ``env`` / ``learning_iteration`` for train."""
  racket_hand: Literal["left", "right"] | None = None
  """G1 tennis: ``left`` / ``right`` racket XML and hit-arm rewards. Unset keeps the task default (left)."""


_DEFAULT_COLLECT_OUTPUT_DIR = "logs/play_dof_pos"

_DEBUG_VIS_EXTRA_ATTRS = (
  "custom_racket_collision_debug_vis",
  "court_debug_vis",
  "court_surface_debug_vis",
  "estimator_debug_vis_traj",
)


def _disable_env_debug_vis(env_cfg) -> None:
  """Turn off command debug overlays (ghost, court lines, collision cylinder, ...)."""
  commands = getattr(env_cfg, "commands", None)
  if not commands:
    return
  for cmd in commands.values():
    if hasattr(cmd, "debug_vis"):
      cmd.debug_vis = False
    for attr in _DEBUG_VIS_EXTRA_ATTRS:
      if hasattr(cmd, attr):
        setattr(cmd, attr, False)
  print("[INFO]: Debug visualization disabled")


def _run_headless_video_recording(env, policy, n_steps: int) -> None:
  """Step the wrapped env until ``VideoRecorder`` finishes, then return."""
  print(f"[INFO] Headless video recording: {n_steps} steps")
  obs, _ = env.reset()
  with torch.inference_mode():
    for i in range(n_steps):
      actions = policy(obs)
      step_out = env.step(actions)
      obs = step_out[0] if isinstance(step_out, tuple) else step_out
      if (i + 1) % 50 == 0 or (i + 1) == n_steps:
        print(f"[INFO] Recorded {i + 1}/{n_steps} frames")


def _wants_per_motion_directory_collection(cfg: PlayConfig) -> bool:
  """True when play should export one replay ``.npz`` per input motion clip."""
  if cfg.motion_directory is None:
    return False
  if cfg.collect_per_motion_directory:
    return True
  out = Path(cfg.collect_output_dir).expanduser().resolve()
  default = Path(_DEFAULT_COLLECT_OUTPUT_DIR).expanduser().resolve()
  return out != default


def _uses_default_collect_motion_window(cfg: PlayConfig) -> bool:
  return abs(cfg.collect_motion_time_start) < 1e-9 and abs(cfg.collect_motion_time_end - 1.0) < 1e-9


def _collect_run_dir(cfg: PlayConfig, task_id: str) -> Path:
  """``<collect_output_dir>/<timestamp>_<task_id>/`` for one collection run."""
  base = Path(cfg.collect_output_dir).expanduser().resolve()
  base.mkdir(parents=True, exist_ok=True)
  safe_task = task_id.replace("/", "_")
  ts = datetime.now().strftime("%Y%m%d_%H%M%S")
  run_dir = base / f"{ts}_{safe_task}"
  run_dir.mkdir(parents=True, exist_ok=True)
  return run_dir


def _make_motion_time_fn(motion_term, env0: torch.Tensor, step_dt: float):
  def _get_motion_time_s() -> float:
    if hasattr(motion_term, "current_time_seconds"):
      return float(motion_term.current_time_seconds(env0)[0].item())
    if hasattr(motion_term, "time_steps") and hasattr(motion_term, "motion_ids"):
      ts = motion_term.time_steps[env0].to(torch.float32)
      mids = motion_term.motion_ids[env0]
      fps_values = getattr(getattr(motion_term, "motion", None), "fps_values", None)
      if fps_values is not None:
        fps = torch.clamp(fps_values[mids], min=1.0)
        return float((ts / fps)[0].item())
      return float((ts * step_dt)[0].item())
    raise RuntimeError(
      f"Unsupported motion command time API: {type(motion_term).__name__}"
    )

  return _get_motion_time_s


def _motion_clip_frame_count(motion_term, env_id: int = 0) -> int:
  mid = int(motion_term.motion_ids[env_id].item())
  return int(motion_term.motion.lengths[mid].item())


def _clip_duration_s(motion_term, env_id: int = 0, *, step_dt: float) -> float:
  """Clip length in seconds on the motion timeline."""
  length = _motion_clip_frame_count(motion_term, env_id)
  motion = motion_term.motion
  mid = int(motion_term.motion_ids[env_id].item())
  fps_values = getattr(motion, "fps_values", None)
  if fps_values is not None:
    fps = float(fps_values[mid].item())
    return max((length - 1) / max(fps, 1.0), 0.0)
  # Tracking: one reference frame per simulation step.
  return max((length - 1) * step_dt, 0.0)


def _uses_tracking_frame_timeline(motion_term) -> bool:
  """Tracking advances integer ``time_steps`` and loops via ``_resample_command``."""
  return hasattr(motion_term, "time_steps") and not hasattr(
    motion_term.motion, "fps_values"
  )


def _collect_time_window_for_clip(
  cfg: PlayConfig, clip_duration_s: float, *, per_motion_directory: bool
) -> tuple[float, float]:
  if per_motion_directory and _uses_default_collect_motion_window(cfg):
    return 0.0, float(clip_duration_s)
  t_start = float(cfg.collect_motion_time_start)
  if cfg.collect_motion_time_end > cfg.collect_motion_time_start:
    t_end = min(float(cfg.collect_motion_time_end), clip_duration_s)
  else:
    t_end = float(clip_duration_s)
  return t_start, t_end


def _downsample_collected_trajectory(
  motion_time: np.ndarray,
  sim_time: np.ndarray,
  dof_pos: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  output_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Resample dense rollout buffers to a uniform ``output_hz`` grid on ``motion_time``."""
  n = int(motion_time.shape[0])
  if n == 0:
    return motion_time, sim_time, dof_pos, root_pos, root_quat
  if output_hz <= 0.0:
    return motion_time, sim_time, dof_pos, root_pos, root_quat

  output_dt = 1.0 / float(output_hz)
  t0 = float(motion_time[0])
  t1 = float(motion_time[-1])
  target_t = np.arange(t0, t1 + 0.5 * output_dt, output_dt, dtype=np.float32)
  if target_t.size == 0:
    target_t = np.asarray([t0], dtype=np.float32)
  if target_t.size == 1 and n > 1:
    target_t = np.asarray([t0, t1], dtype=np.float32)

  idx = np.searchsorted(motion_time, target_t, side="left")
  idx = np.clip(idx, 0, n - 1)
  # Prefer nearest neighbor in time (handles slight uneven spacing from dynamic dt).
  left = np.clip(idx - 1, 0, n - 1)
  pick_left = np.abs(motion_time[left] - target_t) <= np.abs(motion_time[idx] - target_t)
  idx = np.where(pick_left, left, idx)

  dof_out = dof_pos[idx]
  root_pos_out = root_pos[idx]
  root_quat_out = root_quat[idx]
  sim_out = sim_time[idx].astype(np.float32)
  return target_t, sim_out, dof_out, root_pos_out, root_quat_out


def _save_collected_trajectory_npz(
  save_path: Path,
  *,
  dof_pos: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  motion_time: np.ndarray,
  sim_time: np.ndarray,
  sample_hz: float,
  sample_dt: float,
  metadata: dict[str, object],
  joint_names: tuple[str, ...] = (),
) -> None:
  """Write dense replay trajectory at ``sample_hz`` (no resampling)."""
  arrays: dict[str, np.ndarray] = {
    "dof_pos": dof_pos,
    "joint_pos": dof_pos,
    "root_pos": root_pos,
    "root_quat": root_quat,
    "motion_time": motion_time.astype(np.float32),
    "sim_time": sim_time.astype(np.float32),
    "fps": np.asarray([sample_hz], dtype=np.float32),
    "sample_dt": np.asarray([sample_dt], dtype=np.float32),
  }
  for key, val in metadata.items():
    if isinstance(val, (str, Path)):
      arrays[key] = np.asarray(str(val), dtype=object)
    elif isinstance(val, (int, float, np.integer, np.floating)):
      arrays[key] = np.asarray(val, dtype=np.float32)
    else:
      arrays[key] = np.asarray(val, dtype=object)
  if joint_names:
    arrays["joint_names"] = np.asarray(joint_names, dtype=object)
  np.savez_compressed(save_path, **arrays)


def _run_dof_pos_collection(task_id: str, env, policy, cfg: PlayConfig) -> None:
  """Collect robot state in a motion-time window for ``N`` independent play rollouts.

  Per sample: ``dof_pos``, ``root_pos`` (world XYZ), ``root_quat`` (wxyz), plus
  ``motion_time`` / ``sim_time`` timestamps.
  """
  if cfg.collect_dof_pos_n <= 0:
    return
  if cfg.collect_sample_dt <= 0.0:
    raise ValueError(f"collect_sample_dt must be > 0, got {cfg.collect_sample_dt}")
  if cfg.collect_motion_time_end <= cfg.collect_motion_time_start:
    raise ValueError(
      "collect_motion_time_end must be > collect_motion_time_start, got "
      f"{cfg.collect_motion_time_start}..{cfg.collect_motion_time_end}"
    )
  if cfg.collect_max_steps_per_rollout <= 0:
    raise ValueError(
      f"collect_max_steps_per_rollout must be > 0, got {cfg.collect_max_steps_per_rollout}"
    )

  run_dir = _collect_run_dir(cfg, task_id)
  uw = env.unwrapped
  robot = uw.scene["robot"]
  step_dt = float(uw.step_dt)
  env0 = torch.tensor([0], dtype=torch.long, device=uw.device)
  motion_term = uw.command_manager.get_term("motion")
  joint_names = tuple(getattr(robot, "joint_names", ()))
  dense_hz = 1.0 / float(cfg.collect_sample_dt)
  output_hz = float(cfg.collect_output_hz) if cfg.collect_output_hz > 0.0 else dense_hz
  print(
    "[INFO]: DOF collection mode enabled | "
    f"n={cfg.collect_dof_pos_n}, motion_time=[{cfg.collect_motion_time_start}, {cfg.collect_motion_time_end}], "
    f"dense_hz={dense_hz:.1f} (dt={cfg.collect_sample_dt}), "
    f"output_hz={output_hz:.1f}, format=npz, out_dir={run_dir}"
  )

  saved = 0
  attempts = 0
  max_attempts = max(cfg.collect_dof_pos_n * 100, cfg.collect_dof_pos_n)

  _get_motion_time_s = _make_motion_time_fn(motion_term, env0, step_dt)

  with torch.inference_mode():
    while saved < cfg.collect_dof_pos_n:
      attempts += 1
      if attempts > max_attempts:
        raise RuntimeError(
          f"Collection aborted: too many failed attempts ({attempts - 1}), "
          f"saved={saved}/{cfg.collect_dof_pos_n}. Check hit condition / policy."
        )
      print(
        f"[INFO]: Start rollout attempt {attempts} "
        f"(saved {saved}/{cfg.collect_dof_pos_n})"
      )
      obs, _ = env.reset()
      elapsed = 0.0
      next_sample_motion_t = float(cfg.collect_motion_time_start)
      started = False
      reached_window_end = False
      wrapped = False
      prev_motion_t: float | None = None
      data: dict[str, list] = {
        "dof_pos": [],
        "root_pos": [],
        "root_quat": [],
        "motion_time": [],
        "sim_time": [],
      }
      for _ in range(cfg.collect_max_steps_per_rollout):
        step_i = _ + 1
        actions = policy(obs)
        step_out = env.step(actions)
        # RslRlVecEnvWrapper returns 4-tuple in play path; keep compatibility.
        if isinstance(step_out, tuple):
          obs = step_out[0]
        else:
          obs = step_out
        elapsed += step_dt

        motion_t = _get_motion_time_s()
        #print(motion_t)
        if prev_motion_t is not None and started and motion_t + 1e-6 < prev_motion_t:
          wrapped = True
          break
        prev_motion_t = motion_t
        if motion_t >= cfg.collect_motion_time_start:
          started = True
        if started and motion_t > cfg.collect_motion_time_end:
          reached_window_end = True
          break
        if (
          started
          and motion_t <= cfg.collect_motion_time_end
          and motion_t + 1e-9 >= next_sample_motion_t
        ):
          data["dof_pos"].append(
            robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
          )
          data["root_pos"].append(
            robot.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float32)
          )
          data["root_quat"].append(
            robot.data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
          )
          data["motion_time"].append(motion_t)
          data["sim_time"].append(elapsed)
          next_sample_motion_t += float(cfg.collect_sample_dt)

      if wrapped:
        print(f"[INFO]: attempt {attempts} failed (motion timeline wrapped), retrying...")
        continue
      if not reached_window_end:
        print(f"[INFO]: attempt {attempts} failed (window end not reached), retrying...")
        continue

      save_path = run_dir / f"rollout_{saved:03d}.npz"
      motion_time = np.asarray(data["motion_time"], dtype=np.float32)
      sim_time = np.asarray(data["sim_time"], dtype=np.float32)
      dof_pos = (
        np.stack(data["dof_pos"], axis=0)
        if data["dof_pos"]
        else np.zeros((0, 0), dtype=np.float32)
      )
      root_pos = (
        np.stack(data["root_pos"], axis=0)
        if data["root_pos"]
        else np.zeros((0, 3), dtype=np.float32)
      )
      root_quat = (
        np.stack(data["root_quat"], axis=0)
        if data["root_quat"]
        else np.zeros((0, 4), dtype=np.float32)
      )
      n_dense = int(motion_time.shape[0])
      if cfg.collect_output_hz > 0.0 and abs(output_hz - dense_hz) > 1e-3:
        motion_time, sim_time, dof_pos, root_pos, root_quat = _downsample_collected_trajectory(
          motion_time,
          sim_time,
          dof_pos,
          root_pos,
          root_quat,
          output_hz,
        )
      saved_sample_dt = 1.0 / output_hz if cfg.collect_output_hz > 0.0 else float(cfg.collect_sample_dt)
      _save_collected_trajectory_npz(
        save_path,
        dof_pos=dof_pos,
        root_pos=root_pos,
        root_quat=root_quat,
        motion_time=motion_time,
        sim_time=sim_time,
        sample_hz=1.0 / saved_sample_dt,
        sample_dt=float(saved_sample_dt),
        metadata={
          "task_id": task_id,
          "rollout_index": saved,
          "attempt_index": attempts,
          "collect_sample_dt_dense": float(cfg.collect_sample_dt),
          "collect_output_hz": float(output_hz),
          "motion_time_start": float(cfg.collect_motion_time_start),
          "motion_time_end": float(cfg.collect_motion_time_end),
        },
        joint_names=joint_names,
      )
      saved += 1
      n_out = int(motion_time.shape[0])
      ds_note = (
        f", downsampled {n_dense}@{dense_hz:.0f}Hz -> {n_out}@{output_hz:.0f}Hz"
        if n_out != n_dense
        else ""
      )
      print(
        f"[INFO]: Saved rollout {saved}/{cfg.collect_dof_pos_n} -> {save_path} "
        f"(samples={n_out}{ds_note})"
      )


def _run_motion_directory_collection(task_id: str, env, policy, cfg: PlayConfig) -> None:
  """Collect one dense replay ``.npz`` per input clip under ``--motion-directory``."""
  if cfg.collect_sample_dt <= 0.0:
    raise ValueError(f"collect_sample_dt must be > 0, got {cfg.collect_sample_dt}")
  if cfg.collect_max_steps_per_rollout <= 0:
    raise ValueError(
      f"collect_max_steps_per_rollout must be > 0, got {cfg.collect_max_steps_per_rollout}"
    )
  dir_info = _validate_play_motion_directory(cfg)
  if dir_info is None:
    raise ValueError(
      "Per-clip collection requires --motion-directory with at least one .npz file."
    )
  motion_dir, _ = dir_info
  clip_paths = sorted(motion_dir.glob("*.npz"))

  run_dir = _collect_run_dir(cfg, task_id)
  uw = env.unwrapped
  robot = uw.scene["robot"]
  step_dt = float(uw.step_dt)
  env0 = torch.tensor([0], dtype=torch.long, device=uw.device)
  motion_term = uw.command_manager.get_term("motion")
  if not hasattr(motion_term, "swap_reference_motion_file"):
    raise RuntimeError(
      "Per-clip collection requires MotionCommand.swap_reference_motion_file; "
      f"got {type(motion_term).__name__}"
    )
  _get_motion_time_s = _make_motion_time_fn(motion_term, env0, step_dt)
  sample_hz = 1.0 / float(cfg.collect_sample_dt)
  joint_names = tuple(getattr(robot, "joint_names", ()))
  print(
    "[INFO]: Motion-directory collection | "
    f"clips={len(clip_paths)}, sample_hz={sample_hz:.1f} (dt={cfg.collect_sample_dt}), "
    f"format=npz (no resample), out_dir={run_dir}"
  )

  saved = 0
  with torch.inference_mode():
    for clip_i, npz_path in enumerate(clip_paths, start=1):
      stem = npz_path.stem
      save_path = run_dir / f"{stem}.npz"
      print(f"[INFO]: Clip {clip_i}/{len(clip_paths)}: {npz_path.name}")
      motion_term.swap_reference_motion_file(str(npz_path.resolve()))
      n_frames = _motion_clip_frame_count(motion_term, env_id=0)
      clip_duration_s = _clip_duration_s(motion_term, env_id=0, step_dt=step_dt)
      t_start, t_end = _collect_time_window_for_clip(
        cfg, clip_duration_s, per_motion_directory=True
      )
      if t_end <= t_start + 1e-9:
        print(f"[WARN]: Skip {stem} (empty time window [{t_start}, {t_end}])")
        continue

      use_frame_timeline = _uses_tracking_frame_timeline(motion_term)
      obs, _ = env.reset()
      elapsed = 0.0
      next_sample_motion_t = float(t_start)
      started = False
      reached_window_end = False
      early_done = False
      data: dict[str, list] = {
        "dof_pos": [],
        "root_pos": [],
        "root_quat": [],
        "motion_time": [],
        "sim_time": [],
      }

      def _maybe_append_sample(motion_t: float) -> None:
        nonlocal started, next_sample_motion_t
        if motion_t >= t_start:
          started = True
        if (
          started
          and motion_t <= t_end
          and motion_t + 1e-9 >= next_sample_motion_t
        ):
          data["dof_pos"].append(
            robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
          )
          data["root_pos"].append(
            robot.data.root_link_pos_w[0].detach().cpu().numpy().astype(np.float32)
          )
          data["root_quat"].append(
            robot.data.root_link_quat_w[0].detach().cpu().numpy().astype(np.float32)
          )
          data["motion_time"].append(motion_t)
          data["sim_time"].append(elapsed)
          next_sample_motion_t += float(cfg.collect_sample_dt)

      for _ in range(cfg.collect_max_steps_per_rollout):
        frame_idx = int(motion_term.time_steps[env0].item()) if use_frame_timeline else -1
        motion_t = _get_motion_time_s()
        _maybe_append_sample(motion_t)

        if use_frame_timeline:
          # Stop before ``_update_command`` resamples ``time_steps`` back to 0.
          if frame_idx >= n_frames - 1:
            reached_window_end = True
            break
          if started and motion_t > t_end:
            reached_window_end = True
            break
        elif started and motion_t > t_end:
          reached_window_end = True
          break

        actions = policy(obs)
        step_out = env.step(actions)
        if isinstance(step_out, tuple):
          obs = step_out[0]
          dones = step_out[2] if len(step_out) > 2 else None
        else:
          obs = step_out
          dones = None
        elapsed += step_dt

        if dones is not None and bool(dones[0].item()):
          early_done = True
          break

        if not use_frame_timeline:
          motion_t = _get_motion_time_s()
          if started and motion_t > t_end:
            reached_window_end = True
            break

      if early_done:
        print(f"[WARN]: {stem} episode terminated early, skipping.")
        continue
      if not reached_window_end:
        print(
          f"[WARN]: {stem} failed (window [{t_start}, {t_end}] not reached), skipping."
        )
        continue

      motion_time = np.asarray(data["motion_time"], dtype=np.float32)
      sim_time = np.asarray(data["sim_time"], dtype=np.float32)
      dof_pos = (
        np.stack(data["dof_pos"], axis=0)
        if data["dof_pos"]
        else np.zeros((0, 0), dtype=np.float32)
      )
      root_pos = (
        np.stack(data["root_pos"], axis=0)
        if data["root_pos"]
        else np.zeros((0, 3), dtype=np.float32)
      )
      root_quat = (
        np.stack(data["root_quat"], axis=0)
        if data["root_quat"]
        else np.zeros((0, 4), dtype=np.float32)
      )
      n_out = int(motion_time.shape[0])
      _save_collected_trajectory_npz(
        save_path,
        dof_pos=dof_pos,
        root_pos=root_pos,
        root_quat=root_quat,
        motion_time=motion_time,
        sim_time=sim_time,
        sample_hz=sample_hz,
        sample_dt=float(cfg.collect_sample_dt),
        metadata={
          "task_id": task_id,
          "motion_file": str(npz_path.resolve()),
          "motion_stem": stem,
          "clip_index": clip_i - 1,
          "motion_time_start": float(t_start),
          "motion_time_end": float(t_end),
          "clip_duration_s": float(clip_duration_s),
        },
        joint_names=joint_names,
      )
      saved += 1
      print(f"[INFO]: Saved {save_path.name} (samples={n_out} @ {sample_hz:.0f} Hz)")

  print(f"[INFO]: Motion-directory collection done: {saved}/{len(clip_paths)} -> {run_dir}")


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.racket_hand is not None:
    from mjlab.tasks.adapt_tennis.config.g1.env_cfgs import apply_racket_hand

    apply_racket_hand(env_cfg, cfg.racket_hand)
    print(f"[INFO]: Racket hand: {cfg.racket_hand}")

  if cfg.init_state_directory is not None:
    init_state_cmd = (
      env_cfg.commands.get("init_state") if env_cfg.commands is not None else None
    )
    if init_state_cmd is None:
      raise ValueError(
        f"Task {task_id!r} has no commands['init_state']; "
        "--init-state-directory is only for grasp init-state tasks."
      )
    init_state_cmd.state_directory = str(
      Path(cfg.init_state_directory).expanduser().resolve()
    )
    print(f"[INFO]: Using init_state_directory: {init_state_cmd.state_directory}")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  motion_cmd = env_cfg.commands.get("motion") if env_cfg.commands is not None else None
  is_tracking_task = (
    motion_cmd is not None
    and hasattr(motion_cmd, "motion_file")
    and hasattr(motion_cmd, "motion_files")
    and hasattr(motion_cmd, "motion_directory")
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    assert motion_cmd is not None
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    assert motion_cmd is not None

    single = _resolve_play_motion_file(cfg)
    dir_info = (
      _validate_play_motion_directory(cfg) if cfg.motion_directory else None
    )
    if single is not None:
      motion_cmd.motion_file = str(single)
      motion_cmd.motion_directory = ""
      motion_cmd.motion_files = ()
      print(f"[INFO]: Using local motion file: {motion_cmd.motion_file}")
    elif dir_info is not None:
      motion_dir, n_clips = dir_info
      motion_cmd.motion_directory = str(motion_dir)
      motion_cmd.motion_file = ""
      motion_cmd.motion_files = ()
      print(
        f"[INFO]: Using motion_directory ({n_clips} clips); "
        "MotionCommand will resample across clips when each ends."
      )
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --motion-directory /path/to/dir (all ``*.npz``, multi-clip)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None or cfg.motion_directory is not None:
        raise FileNotFoundError(
          "Could not resolve local motion: check ``--motion-file`` or ``--motion-directory``."
        )
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require ``--motion-file`` / ``--motion-directory`` when using "
            "``checkpoint_file`` without ``wandb_run_path``, or provide ``wandb_run_path`` "
            "so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  onnx_path: Path | None = None
  if TRAINED_MODE and cfg.onnx_file is not None:
    onnx_path = Path(cfg.onnx_file).expanduser().resolve()
    if not onnx_path.is_file():
      raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    print(f"[INFO]: Using ONNX policy: {onnx_path.name}")
  if TRAINED_MODE:
    if onnx_path is None:
      log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
      if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
          raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading checkpoint: {resume_path.name}")
      else:
        if cfg.wandb_run_path is None:
          raise ValueError(
            "`wandb_run_path` is required when `checkpoint_file` is not provided."
          )
        resume_path, was_cached = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
        )
        # Extract run_id and checkpoint name from path for display.
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
        )
      log_dir = resume_path.parent
    else:
      log_dir = onnx_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  want_video = TRAINED_MODE and (cfg.video or cfg.video_headless)
  if want_video:
    # Offscreen video on a machine without DISPLAY needs EGL, not GLFW.
    os.environ.setdefault("MUJOCO_GL", "egl")
  render_mode = "rgb_array" if want_video else None
  if (cfg.video or cfg.video_headless) and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  if cfg.motion_warmup_steps is not None:
    apply_motion_reference_warmup(
      env_cfg,
      warmup_steps=int(cfg.motion_warmup_steps),
      step_source=cfg.motion_warmup_step_source or "episode",
    )
  if not cfg.debug_vis:
    _disable_env_debug_vis(env_cfg)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  if not cfg.debug_vis:
    env.update_visualizers = lambda visualizer: None  # type: ignore[method-assign]

  video_length = cfg.video_length
  if want_video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    if cfg.video_duration_s is not None:
      video_length = max(
        1, int(np.ceil(float(cfg.video_duration_s) / float(env.unwrapped.step_dt)))
      )
      print(
        f"[INFO] video_duration_s={cfg.video_duration_s:.3f}s "
        f"(step_dt={env.unwrapped.step_dt:.4f}s) -> video_length={video_length}"
      )
    video_folder = (
      Path(cfg.video_folder).expanduser().resolve()
      if cfg.video_folder
      else log_dir / "videos" / "play"
    )
    env = VideoRecorder(
      env,
      video_folder=video_folder,
      step_trigger=lambda step: step == 0,
      video_length=video_length,
      disable_logger=False,
    )
    print(f"[INFO] Video folder: {video_folder}")

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    if onnx_path is not None:
      import onnxruntime as ort

      sess = ort.InferenceSession(
        str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
      )
      in_names = [x.name for x in sess.get_inputs()]
      out_names = [x.name for x in sess.get_outputs()]
      if not out_names:
        raise RuntimeError(f"ONNX has no outputs: {onnx_path}")
      action_out = "actions" if "actions" in out_names else out_names[0]
      obs_input_name = next((n for n in in_names if "obs" in n.lower()), in_names[0])
      obs_input_shape = next(
        inp.shape for inp in sess.get_inputs() if inp.name == obs_input_name
      )
      expected_obs_dim = (
        int(obs_input_shape[1])
        if len(obs_input_shape) >= 2 and isinstance(obs_input_shape[1], int)
        else None
      )

      class PolicyOnnx:
        def _collect_leaf_tensors(
          self, x, prefix: str = ""
        ) -> list[tuple[str, torch.Tensor]]:
          if isinstance(x, torch.Tensor):
            return [(prefix or "<root>", x)]
          if isinstance(x, np.ndarray):
            return [(prefix or "<root>", torch.from_numpy(x))]
          if isinstance(x, Mapping) or hasattr(x, "items"):
            out: list[tuple[str, torch.Tensor]] = []
            for k, v in x.items():
              key = str(k)
              child_prefix = f"{prefix}.{key}" if prefix else key
              out.extend(self._collect_leaf_tensors(v, child_prefix))
            return out
          return []

        def _flatten_actor_obs(self, x) -> torch.Tensor:
          if isinstance(x, torch.Tensor):
            return x
          if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
          if isinstance(x, Mapping) or hasattr(x, "items"):
            parts: list[torch.Tensor] = []
            for _, v in x.items():
              t = self._flatten_actor_obs(v)
              if t.ndim == 1:
                t = t.unsqueeze(0)
              parts.append(t.reshape(t.shape[0], -1))
            if not parts:
              raise RuntimeError("Empty actor observation dict cannot be flattened.")
            return torch.cat(parts, dim=-1)
          raise TypeError(f"Unsupported ONNX observation type: {type(x)!r}")

        def __call__(self, obs) -> torch.Tensor:
          actor_obs = obs["actor"] if isinstance(obs, dict) else obs
          actor_obs_t: torch.Tensor
          leaves = self._collect_leaf_tensors(actor_obs)
          if expected_obs_dim is not None and leaves:
            selected: torch.Tensor | None = None
            # Prefer semantically obvious keys first.
            for key, t in leaves:
              if key.split(".")[-1].lower() in {"obs", "policy_obs", "actor_obs"}:
                tt = t.unsqueeze(0) if t.ndim == 1 else t
                if tt.reshape(tt.shape[0], -1).shape[1] == expected_obs_dim:
                  selected = tt
                  break
            if selected is None:
              for _, t in leaves:
                tt = t.unsqueeze(0) if t.ndim == 1 else t
                if tt.reshape(tt.shape[0], -1).shape[1] == expected_obs_dim:
                  selected = tt
                  break
            if selected is not None:
              actor_obs_t = selected.reshape(selected.shape[0], -1)
            else:
              actor_obs_t = self._flatten_actor_obs(actor_obs)
          else:
            actor_obs_t = self._flatten_actor_obs(actor_obs)
          x = actor_obs_t.detach().cpu().numpy().astype(np.float32)
          if expected_obs_dim is not None and x.shape[1] != expected_obs_dim:
            leaf_shapes = ", ".join(
              f"{k}:{tuple((v.unsqueeze(0) if v.ndim == 1 else v).reshape((v.unsqueeze(0) if v.ndim == 1 else v).shape[0], -1).shape)}"
              for k, v in leaves
            )
            raise RuntimeError(
              f"ONNX obs dim mismatch: got {x.shape[1]}, expected {expected_obs_dim}. "
              f"Available leaf tensors: {leaf_shapes}"
            )
          feeds: dict[str, np.ndarray] = {}
          for name in in_names:
            lname = name.lower()
            if "obs" in lname:
              feeds[name] = x
            elif "time" in lname:
              feeds[name] = np.zeros((x.shape[0], 1), dtype=np.float32)
            else:
              feeds[name] = x
          out = sess.run([action_out], feeds)[0]
          return torch.as_tensor(out, device=env.unwrapped.device, dtype=torch.float32)

      policy = PolicyOnnx()
    else:
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(env, asdict(agent_cfg), device=device)
      runner.load(
        str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
      )
      if cfg.explore_jit:
        if resume_path is None:
          raise RuntimeError("explore_jit requires a local checkpoint_file.")
        jit_model = torch.jit.script(runner.alg.get_policy().as_jit())
        jit_dir = resume_path.parent / "jit"
        jit_dir.mkdir(parents=True, exist_ok=True)
        suffix = resume_path.stem.replace("model_", "")
        jit_path = jit_dir / f"modeljit_{suffix}.pt"
        jit_model.save(str(jit_path))
        print(f"[INFO]: Exported JIT policy to: {jit_path}")
      policy = runner.get_inference_policy(device=device)

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None and onnx_path is None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  if _wants_per_motion_directory_collection(cfg):
    _run_motion_directory_collection(task_id, env, policy, cfg)
    env.close()
    return
  if cfg.collect_dof_pos_n > 0:
    _run_dof_pos_collection(task_id, env, policy, cfg)
    env.close()
    return
  if want_video and cfg.video_headless:
    _run_headless_video_recording(env, policy, video_length)
    env.close()
    return

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    viser_server = None
    if cfg.viser_port is not None:
      import viser

      viser_server = viser.ViserServer(port=int(cfg.viser_port), label="mjlab")
      print(f"[INFO]: Viser server listening on port {cfg.viser_port}")
    ViserPlayViewer(
      env,
      policy,
      viser_server=viser_server,
      checkpoint_manager=ckpt_manager,
    ).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
