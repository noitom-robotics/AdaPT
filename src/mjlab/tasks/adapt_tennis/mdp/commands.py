from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import torch.nn.functional as F

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.tasks.adapt_tennis.replay_joint_names import (
  G1_REPLAY_JOINT_NAMES_27,
)
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _batched_quat_slerp(
  q0: torch.Tensor, q1: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
  """Batched spherical linear interpolation between two quaternion fields.

  Args:
    q0, q1: ``(N, ..., 4)`` quaternions in ``(w, x, y, z)`` ordering.
    alpha: ``(N,)`` interpolation coefficient in ``[0, 1]``. Broadcasts over
      the trailing per-body axes of ``q0`` / ``q1``.

  Falls back to linear interpolation + renormalization when the two
  quaternions are extremely close, which avoids divide-by-zero in ``sin(0)``.
  """
  q1 = q1.clone()
  dot = (q0 * q1).sum(dim=-1, keepdim=True)
  sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
  q1 = q1 * sign
  dot = dot * sign
  dot = dot.clamp(-1.0, 1.0)

  trailing = (1,) * (q0.ndim - 1)
  alpha_b = alpha.view(-1, *trailing)

  angle = torch.acos(dot)
  sin_angle = torch.sin(angle)
  # If the two quats are nearly parallel, fall back to linear interpolation.
  small = sin_angle.abs() < 1.0e-6
  w0 = torch.where(small, 1.0 - alpha_b, torch.sin((1.0 - alpha_b) * angle) / sin_angle.clamp_min(1.0e-12))
  w1 = torch.where(small, alpha_b, torch.sin(alpha_b * angle) / sin_angle.clamp_min(1.0e-12))
  out = w0 * q0 + w1 * q1
  out = out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(1.0e-12)
  return out


AlignHeadingFrame = Literal["first", "last", "none"]


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Extract yaw (Z) from quaternion in wxyz layout."""
  w = quat_wxyz[..., 0]
  x = quat_wxyz[..., 1]
  y = quat_wxyz[..., 2]
  z = quat_wxyz[..., 3]
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return torch.atan2(siny_cosp, cosy_cosp)


def _align_motion_heading_to_frame(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  *,
  root_body_index: int,
  ref_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
  """Rotate the full clip so root yaw at ``ref_frame`` becomes zero (+X forward).

  Positions are rotated about the first-frame root position (same pivot convention
  as ``gmr_pkl_sim_to_npz``). Linear / angular velocities are rotated by the
  same world yaw.

  Returns:
    Transformed body tensors, plus ``(applied_yaw_rad, pivot_xyz)`` describing the
    rigid yaw used (so cfg ``toss_pos`` / ``hit_pos`` can be transformed the same way).
  """
  t_frames = int(body_pos_w.shape[0])
  pivot = body_pos_w[0, root_body_index].detach().clone()
  if t_frames < 1:
    return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, 0.0, pivot
  ref_frame = int(max(0, min(ref_frame, t_frames - 1)))
  yaw_ref = _yaw_from_quat_wxyz(body_quat_w[ref_frame, root_body_index])
  applied_yaw = float((-yaw_ref).item())
  if abs(applied_yaw) < 1.0e-8:
    return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, 0.0, pivot

  half = 0.5 * torch.tensor(applied_yaw, device=body_quat_w.device, dtype=body_quat_w.dtype)
  q_bias = torch.stack(
    (
      torch.cos(half),
      torch.zeros((), device=body_quat_w.device, dtype=body_quat_w.dtype),
      torch.zeros((), device=body_quat_w.device, dtype=body_quat_w.dtype),
      torch.sin(half),
    ),
    dim=-1,
  )
  q_bias_b = q_bias.view(1, 1, 4).expand_as(body_quat_w)
  body_quat_corr = quat_mul(q_bias_b, body_quat_w)
  body_quat_corr = body_quat_corr / torch.linalg.norm(
    body_quat_corr, dim=-1, keepdim=True
  ).clamp_min(1.0e-8)

  center = body_pos_w[0:1, root_body_index : root_body_index + 1, :].clone()
  rel = body_pos_w - center
  body_pos_corr = center + quat_apply(q_bias_b, rel)
  body_lin_corr = quat_apply(q_bias_b, body_lin_vel_w)
  body_ang_corr = quat_apply(q_bias_b, body_ang_vel_w)
  return body_pos_corr, body_quat_corr, body_lin_corr, body_ang_corr, applied_yaw, pivot


def _root_body_index(target_body_names: tuple[str, ...] | None, n_bodies: int) -> int:
  if target_body_names:
    for i, name in enumerate(target_body_names):
      if name == "pelvis" or name.endswith("/pelvis"):
        return i
  return 0 if n_bodies > 0 else 0


def _npz_float(value: object, *, name: str = "value") -> float:
  """Convert npz scalars, including length-1 arrays such as ``fps`` ``(1,)``."""
  arr = np.asarray(value)
  if arr.size != 1:
    raise ValueError(f"Expected a scalar {name}, got shape {arr.shape}")
  return float(arr.reshape(-1)[0])


class MotionLoader:
  """Loads one motion clip from ``.npz``.

  Column mapping strategy
  -----------------------
  When the ``.npz`` carries ``body_names`` / ``joint_names`` arrays (the
  self-describing format produced by ``gmr_pkl_sim_to_npz.py`` and
  ``csv_to_npz.py``), columns are mapped to the runtime robot **by name**:

  - ``body_*_w`` columns are looked up against ``target_body_names`` (= the
    ``MotionCommand`` ``cfg.body_names`` tracked subset).
  - ``joint_pos`` / ``joint_vel`` columns are looked up against
    ``target_joint_names`` (= ``G1_REPLAY_JOINT_NAMES_27`` for AdaPT Tennis tracking).

  Name-based mapping is robust to XML edits that insert/move bodies or joints:
  the npz never has to be regenerated when the robot XML changes, as long as
  the tracked names exist.

  Legacy npz (without ``body_names`` / ``joint_names``) fall back to the older
  index-based heuristic and emit a warning. The old behavior:

  - ``body_indexes`` are indices into the runtime robot's body array. Some
    npz store ``body_*`` with one row per *tracked* body (same order as
    ``cfg.body_names``); in that compact layout use columns ``0..N-1``.
    Otherwise the npz is assumed to be full-layout with one column per body in
    the npz-generator XML's order, indexed by ``body_indexes`` (fragile when
    that XML diverges from the runtime XML).
  """

  def __init__(
    self,
    motion_file: str,
    body_indexes: torch.Tensor,
    device: str = "cpu",
    *,
    target_body_names: tuple[str, ...] | None = None,
    target_joint_names: tuple[str, ...] | None = None,
    align_heading_to_frame: AlignHeadingFrame = "last",
  ) -> None:
    data = np.load(motion_file, allow_pickle=True)
    self.fps = _npz_float(data["fps"], name="fps") if "fps" in data else 30.0
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    raw_joint_pos = torch.tensor(
      data["joint_pos"], dtype=torch.float32, device=device
    )
    raw_joint_vel = torch.tensor(
      data["joint_vel"], dtype=torch.float32, device=device
    )

    npz_body_names = self._read_names(data, "body_names")
    npz_joint_names = self._read_names(data, "joint_names")
    self.npz_body_names = npz_body_names
    self.npz_joint_names = npz_joint_names

    # --- body column mapping ---
    if npz_body_names is not None and target_body_names is not None:
      missing = [n for n in target_body_names if n not in npz_body_names]
      if missing:
        raise ValueError(
          f"Motion file {motion_file!r} is missing body columns for "
          f"{missing}. npz body_names = {npz_body_names}."
        )
      motion_body_cols = torch.tensor(
        [npz_body_names.index(n) for n in target_body_names],
        dtype=torch.long,
        device=device,
      )
    else:
      if target_body_names is not None and npz_body_names is None:
        print(
          f"[MotionLoader] WARNING: {motion_file!r} has no ``body_names`` "
          "array; falling back to index-based body mapping. Regenerate the "
          "motion via gmr_pkl_sim_to_npz / csv_to_npz to get a "
          "self-describing npz that is robust to XML changes."
        )
      bi = body_indexes.to(device=device)
      n_body_motion = int(self._body_pos_w.shape[1])
      if bi.numel() and int(bi.max().item()) >= n_body_motion:
        motion_body_cols = torch.arange(
          bi.shape[0], dtype=torch.long, device=device
        )
        assert len(motion_body_cols) == len(body_indexes)
      else:
        motion_body_cols = bi
    self._body_indexes = motion_body_cols
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]

    # --- joint column mapping ---
    if npz_joint_names is not None and target_joint_names is not None:
      missing_j = [n for n in target_joint_names if n not in npz_joint_names]
      if missing_j:
        raise ValueError(
          f"Motion file {motion_file!r} is missing joint columns for "
          f"{missing_j}. npz joint_names = {npz_joint_names}."
        )
      joint_cols = torch.tensor(
        [npz_joint_names.index(n) for n in target_joint_names],
        dtype=torch.long,
        device=device,
      )
      self._joint_cols = joint_cols
      self.joint_pos = raw_joint_pos[:, joint_cols]
      self.joint_vel = raw_joint_vel[:, joint_cols]
    else:
      if target_joint_names is not None and npz_joint_names is None:
        n_target = len(target_joint_names)
        if raw_joint_pos.shape[1] != n_target:
          raise ValueError(
            f"Motion file {motion_file!r} has no ``joint_names`` array and its "
            f"joint_pos column count {raw_joint_pos.shape[1]} != target joint "
            f"count {n_target}. Regenerate the motion to embed joint_names, "
            "or fix the motion source."
          )
        print(
          f"[MotionLoader] WARNING: {motion_file!r} has no ``joint_names`` "
          "array; assuming joint_pos columns already follow target order. "
          "Regenerate the motion to get a self-describing npz."
        )
      self._joint_cols = torch.arange(
        raw_joint_pos.shape[1], dtype=torch.long, device=device
      )
      self.joint_pos = raw_joint_pos
      self.joint_vel = raw_joint_vel

    self.time_step_total = self.joint_pos.shape[0]
    self.heading_align_yaw_rad: float = 0.0
    self.heading_align_pivot_w = torch.zeros(
      3, dtype=torch.float32, device=device
    )
    if int(self.body_pos_w.shape[0]) > 0 and int(self.body_pos_w.shape[1]) > 0:
      root_idx0 = _root_body_index(target_body_names, int(self.body_pos_w.shape[1]))
      self.heading_align_pivot_w = self.body_pos_w[0, root_idx0].detach().clone()

    # Serve clips: align *last* frame heading to +X so the start can face
    # sideways (realistic serve stance). Override via ``align_heading_to_frame``.
    if align_heading_to_frame != "none" and self.time_step_total > 0:
      root_idx = _root_body_index(target_body_names, int(self.body_pos_w.shape[1]))
      ref_frame = (
        0 if align_heading_to_frame == "first" else self.time_step_total - 1
      )
      yaw_before = float(
        _yaw_from_quat_wxyz(self.body_quat_w[ref_frame, root_idx]).item()
      )
      (
        self.body_pos_w,
        self.body_quat_w,
        self.body_lin_vel_w,
        self.body_ang_vel_w,
        self.heading_align_yaw_rad,
        self.heading_align_pivot_w,
      ) = _align_motion_heading_to_frame(
        self.body_pos_w,
        self.body_quat_w,
        self.body_lin_vel_w,
        self.body_ang_vel_w,
        root_body_index=root_idx,
        ref_frame=ref_frame,
      )
      yaw_after = float(
        _yaw_from_quat_wxyz(self.body_quat_w[ref_frame, root_idx]).item()
      )
      print(
        f"[MotionLoader] align_heading_to_frame={align_heading_to_frame!r} "
        f"ref_frame={ref_frame} yaw {yaw_before:+.3f} -> {yaw_after:+.3f} rad "
        f"(applied {self.heading_align_yaw_rad:+.3f}) ({motion_file!r})"
      )

  @staticmethod
  def _read_names(data: "np.lib.npyio.NpzFile", key: str) -> tuple[str, ...] | None:
    if key not in data.files:
      return None
    arr = data[key]
    return tuple(str(x) for x in arr.tolist())


class MultiMotionLoader:
  """Holds multiple ``MotionLoader`` clips; gather by per-env ``motion_ids`` + ``time_steps``."""

  def __init__(
    self,
    motion_paths: tuple[str, ...],
    body_indexes: torch.Tensor,
    device: str,
    *,
    target_body_names: tuple[str, ...] | None = None,
    target_joint_names: tuple[str, ...] | None = None,
    align_heading_to_frame: AlignHeadingFrame = "last",
  ) -> None:
    if not motion_paths:
      raise ValueError("MultiMotionLoader requires at least one motion path.")
    self.motions = [
      MotionLoader(
        p,
        body_indexes,
        device=device,
        target_body_names=target_body_names,
        target_joint_names=target_joint_names,
        align_heading_to_frame=align_heading_to_frame,
      )
      for p in motion_paths
    ]
    self.num_motions = len(self.motions)
    ref = self.motions[0]
    for i, m in enumerate(self.motions[1:], start=1):
      if m.joint_pos.shape[1] != ref.joint_pos.shape[1]:
        raise ValueError(
          f"Motion {i} joint count {m.joint_pos.shape[1]} != first motion "
          f"{ref.joint_pos.shape[1]} ({motion_paths[0]} vs {motion_paths[i]})."
        )
      if m.body_pos_w.shape[1:] != ref.body_pos_w.shape[1:]:
        raise ValueError(
          f"Motion {i} body tensor shape {m.body_pos_w.shape} != "
          f"first motion {ref.body_pos_w.shape}."
        )
      # When self-describing, also verify column-name alignment (catches mixed
      # generator XMLs early instead of letting subtle errors propagate).
      if (
        ref.npz_body_names is not None
        and m.npz_body_names is not None
        and ref.npz_body_names != m.npz_body_names
      ):
        raise ValueError(
          f"Motion {i} body_names differ from motion 0 "
          f"({motion_paths[0]} vs {motion_paths[i]}). After name-based "
          "remap the shapes match, but the source XMLs disagree on body order; "
          "regenerate all clips with the same robot XML."
        )
      if (
        ref.npz_joint_names is not None
        and m.npz_joint_names is not None
        and ref.npz_joint_names != m.npz_joint_names
      ):
        raise ValueError(
          f"Motion {i} joint_names differ from motion 0 "
          f"({motion_paths[0]} vs {motion_paths[i]})."
        )
    self.device = device
    self.time_step_totals = torch.tensor(
      [m.time_step_total for m in self.motions],
      dtype=torch.long,
      device=device,
    )
    self.fps_values = torch.tensor(
      [float(m.fps) for m in self.motions],
      dtype=torch.float32,
      device=device,
    )
    self.max_time_step_total = int(self.time_step_totals.max().item())

    # Padded stacks (M, T_max, …) so ``_gather`` is one GPU index op per call instead of
    # O(num_motions) Python loops (which made multi-clip training much slower than one clip).
    max_t = self.max_time_step_total
    self._stacked: dict[str, torch.Tensor] = {}
    for attr in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      first = getattr(ref, attr)
      trailing = first.shape[1:]
      buf = torch.zeros(
        self.num_motions,
        max_t,
        *trailing,
        dtype=first.dtype,
        device=device,
      )
      for i, m in enumerate(self.motions):
        src = getattr(m, attr)
        ti = int(src.shape[0])
        buf[i, :ti] = src
        if ti < max_t:
          buf[i, ti:] = src[-1]
      self._stacked[attr] = buf

    self.heading_align_yaw_rad = torch.tensor(
      [float(m.heading_align_yaw_rad) for m in self.motions],
      dtype=torch.float32,
      device=device,
    )
    self.heading_align_pivot_w = torch.stack(
      [m.heading_align_pivot_w.to(device=device, dtype=torch.float32) for m in self.motions],
      dim=0,
    )

  # ``MotionLoader``-shaped views of the first clip (for ONNX export and any code that
  # expects a single stacked trajectory, e.g. ``_OnnxMotionModel``).
  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motions[0].joint_pos

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motions[0].joint_vel

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self.motions[0].body_pos_w

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motions[0].body_quat_w

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motions[0].body_lin_vel_w

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motions[0].body_ang_vel_w

  def gather_joint_pos(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self._gather("joint_pos", motion_ids, time_steps)

  def gather_joint_vel(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self._gather("joint_vel", motion_ids, time_steps)

  def gather_body_pos_w(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self._gather("body_pos_w", motion_ids, time_steps)

  def gather_body_quat_w(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self._gather("body_quat_w", motion_ids, time_steps)

  def gather_body_lin_vel_w(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self._gather("body_lin_vel_w", motion_ids, time_steps)

  def gather_body_ang_vel_w(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    return self._gather("body_ang_vel_w", motion_ids, time_steps)

  def _gather(
    self,
    attr: str,
    motion_ids: torch.Tensor,
    time_steps: torch.Tensor,
  ) -> torch.Tensor:
    stacked = self._stacked[attr]
    max_t_idx = torch.clamp(self.time_step_totals[motion_ids] - 1, min=0)
    tt = torch.minimum(torch.clamp(time_steps, min=0), max_t_idx)
    return stacked[motion_ids, tt]

  # ----------------------------------------------------------------------
  # Float-time interpolated gather (used when ``dynamic_dt_enabled``).
  # ----------------------------------------------------------------------
  def _gather_interp(
    self,
    attr: str,
    motion_ids: torch.Tensor,
    t_seconds: torch.Tensor,
  ) -> torch.Tensor:
    """Linear (or slerp for ``body_quat_w``) interpolation between adjacent frames.

    ``t_seconds`` is per-env float time on the motion timeline. Out-of-range
    samples are clamped to the clip endpoints (consistent with ``_gather``).
    """
    stacked = self._stacked[attr]
    fps = torch.clamp(self.fps_values[motion_ids], min=1.0)
    max_idx = torch.clamp(self.time_step_totals[motion_ids] - 1, min=0)

    frame_f = (t_seconds * fps).clamp(min=0.0)
    f0_long = frame_f.floor().to(torch.long)
    f0 = torch.minimum(f0_long, max_idx)
    f1 = torch.minimum(f0 + 1, max_idx)
    alpha = (frame_f - f0_long.to(frame_f.dtype)).clamp(0.0, 1.0)
    # When clamped at the tail (f0 == max_idx), force alpha to 0 so we return
    # the last valid sample instead of blending into the held repeat.
    at_tail = f0 >= max_idx
    alpha = torch.where(at_tail, torch.zeros_like(alpha), alpha)

    a0 = stacked[motion_ids, f0]
    a1 = stacked[motion_ids, f1]

    if attr == "body_quat_w":
      # Batched slerp over the per-body quaternion dim.
      return _batched_quat_slerp(a0, a1, alpha)
    # Linear blend for everything else (pos, lin_vel, ang_vel, joint_pos/vel).
    trailing = (1,) * (a0.ndim - 1)
    alpha_b = alpha.view(-1, *trailing)
    return a0 * (1.0 - alpha_b) + a1 * alpha_b

  @property
  def time_step_total(self) -> int:
    """Largest clip length (used by GUI scrubber range)."""
    return self.max_time_step_total

  @property
  def lengths(self) -> torch.Tensor:
    """Alias for :attr:`time_step_totals` (compat with call sites)."""
    return self.time_step_totals

  @property
  def max_time_steps(self) -> int:
    """Alias for :attr:`max_time_step_total` (compat with call sites)."""
    return self.max_time_step_total

  def gather_time_seconds(
    self, motion_ids: torch.Tensor, time_steps: torch.Tensor
  ) -> torch.Tensor:
    fps = torch.clamp(self.fps_values[motion_ids], min=1.0)
    return time_steps.to(dtype=torch.float32) / fps


def _resolve_motion_paths(cfg: MotionCommandCfg) -> tuple[str, ...]:
  """Resolve ``motion_files`` > ``motion_file`` > ``motion_directory``."""
  if cfg.motion_files:
    out: list[str] = []
    for p in cfg.motion_files:
      rp = Path(p).expanduser().resolve()
      if not rp.is_file():
        raise FileNotFoundError(f"motion_files entry not found: {rp}")
      out.append(str(rp))
    return tuple(out)
  if cfg.motion_file:
    p = Path(cfg.motion_file).expanduser().resolve()
    if not p.is_file():
      raise FileNotFoundError(f"motion_file not found: {p}")
    return (str(p),)
  if cfg.motion_directory:
    d = Path(cfg.motion_directory).expanduser().resolve()
    if not d.is_dir():
      raise ValueError(f"motion_directory is not a directory: {d}")
    files = sorted(x for x in d.glob("*.npz") if x.is_file())
    if not files:
      raise ValueError(f"No .npz files under motion_directory: {d}")
    return tuple(str(f) for f in files)
  raise ValueError(
    "MotionCommandCfg requires non-empty ``motion_files``, ``motion_file``, or ``motion_directory``."
  )


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)

    # Map each motion joint column (``G1_REPLAY_JOINT_NAMES_27`` ordering) to its
    # corresponding robot joint index. Extra robot joints that are not in the motion
    # clip are left unmapped; a naive ``torch.arange`` would misalign columns.
    robot_joint_names = list(self.robot.joint_names)
    missing = [n for n in G1_REPLAY_JOINT_NAMES_27 if n not in robot_joint_names]
    if missing:
      raise ValueError(
        "Motion joint names missing from robot.joint_names: "
        f"{missing}. Robot has joints {robot_joint_names}."
      )
    self._motion_to_robot_joint_ids = torch.tensor(
      [robot_joint_names.index(n) for n in G1_REPLAY_JOINT_NAMES_27],
      dtype=torch.long,
      device=self.device,
    )
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    paths = _resolve_motion_paths(cfg)
    self.motion = MultiMotionLoader(
      paths,
      self.body_indexes,
      device=self.device,
      target_body_names=tuple(self.cfg.body_names),
      target_joint_names=G1_REPLAY_JOINT_NAMES_27,
      align_heading_to_frame=self.cfg.align_heading_to_frame,
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # Monotonic timeline for termination accounting (not clamped at clip end).
    self.time_steps_progress = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    # Dynamic-dt state: per-env float timeline (seconds), pending per-step
    # ``dt`` advance, and the ``delta_t`` chosen by the policy last step
    # (used as the ``motion_dt_prev`` observation; ``0`` on reset).
    self._fixed_dt: float = float(self.cfg.fixed_dt)
    self._time_seconds_progress = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._pending_dt = torch.full(
      (self.num_envs,), self._fixed_dt, dtype=torch.float32, device=self.device
    )
    self.last_dt = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    if self.motion.num_motions > 1:
      self.motion_ids[:] = torch.randint(
        0, self.motion.num_motions, (self.num_envs,), device=self.device
      )
      cl = self.motion.lengths[self.motion_ids].float()
      self.time_steps[:] = torch.minimum(
        (torch.rand(self.num_envs, device=self.device) * cl).long(),
        self.motion.lengths[self.motion_ids] - 1,
      )
      self.time_steps_progress[:] = self.time_steps
    self._sync_time_seconds_from_steps()
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0
    # Per-env reset randomization offsets (used by downstream terms).
    self.root_pos_rand_offset_w = torch.zeros(
      self.num_envs, 3, dtype=torch.float32, device=self.device
    )
    self.root_ori_rand_offset_w = torch.zeros(
      self.num_envs, 4, dtype=torch.float32, device=self.device
    )
    self.root_ori_rand_offset_w[:, 0] = 1.0

    self._bin_counts = torch.tensor(
      [
        max(int(m.time_step_total // (1 / env.step_dt)) + 1, 1)
        for m in self.motion.motions
      ],
      dtype=torch.long,
      device=self.device,
    )
    self._max_bin_count = int(self._bin_counts.max().item())
    # Single-motion layout matches legacy: one row, shape (1, B).
    self.bin_failed_count = torch.zeros(
      self.motion.num_motions, self._max_bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    self.bin_count = self._max_bin_count
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._dump_step = 0
    self._dump_txt_path = cfg.dump_obs_txt_path.strip()
    if self._dump_txt_path:
      parent = os.path.dirname(self._dump_txt_path)
      if parent:
        os.makedirs(parent, exist_ok=True)
      with open(self._dump_txt_path, "w", encoding="utf-8") as f:
        f.write("# step env_id actor_obs_values...\n")

  def _motion_warmup_training_step(self) -> int:
    """Global training progress used by :attr:`MotionCommandCfg.motion_warmup_steps`."""
    if self.cfg.motion_warmup_step_source == "learning_iteration":
      it = getattr(self._env, "current_learning_iteration", 0)
      if isinstance(it, torch.Tensor):
        return int(it.item())
      return int(it)
    return int(getattr(self._env, "common_step_counter", 0))

  def _motion_warmup_mask(self) -> torch.Tensor:
    """Per-env mask: reference timeline is held at frame 0 where True."""
    warmup = int(self.cfg.motion_warmup_steps)
    if warmup <= 0:
      return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    src = self.cfg.motion_warmup_step_source
    if src == "episode":
      return self._env.episode_length_buf <= warmup
    if src == "learning_iteration":
      step = self._motion_warmup_training_step()
    else:
      step = int(getattr(self._env, "common_step_counter", 0))
    active = step <= warmup
    return torch.full((self.num_envs,), active, dtype=torch.bool, device=self.device)

  def is_motion_warmup_active(self) -> bool:
    """True if any env is still in the motion-reference warmup window."""
    return bool(torch.any(self._motion_warmup_mask()).item())

  def _freeze_motion_timeline_at_start(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> None:
    """Hold motion reference at frame 0 (``motion_time = 0``) for selected envs."""
    if env_ids is None:
      sel = slice(None)
    else:
      sel = env_ids
    self.time_steps[sel] = 0
    self.time_steps_progress[sel] = 0
    self._time_seconds_progress[sel] = 0.0
    self.last_dt[sel] = 0.0
    if self.cfg.dynamic_dt_enabled:
      self._pending_dt[sel] = self._fixed_dt

  def _sync_time_seconds_from_steps(
    self, env_ids: torch.Tensor | None = None
  ) -> None:
    """Mirror integer ``time_steps`` into the float ``_time_seconds_progress`` timeline.

    Called whenever the integer index is rewound by resample / start curriculum /
    GUI scrubber so that subsequent dynamic-``dt`` advances continue from a
    consistent float time. Also clears the pending-``dt`` and ``last_dt``
    buffers so the next observation sees ``0`` for the previous-step ``dt``
    (matching "no previous dt" at episode boundary).
    """
    if env_ids is None:
      sel = slice(None)
      ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    else:
      sel = env_ids
      ids_tensor = env_ids
    fps = torch.clamp(self.motion.fps_values[self.motion_ids[ids_tensor]], min=1.0)
    self._time_seconds_progress[sel] = self.time_steps[ids_tensor].to(torch.float32) / fps
    self._pending_dt[sel] = self._fixed_dt
    self.last_dt[sel] = 0.0

  def set_pending_dt(
    self,
    dt_seconds: torch.Tensor,
    delta_t: torch.Tensor | None = None,
  ) -> None:
    """Write the next motion-timeline advance (seconds).

    ``dt_seconds``: shape ``(num_envs,)`` absolute per-step advance (already
    ``fixed_dt + delta_t`` with any dead-band applied).
    ``delta_t``: matching ``(num_envs,)`` residual; cached as
    :attr:`last_dt` so the next observation can expose it via
    ``motion_dt_prev`` (defaults to ``dt_seconds - fixed_dt``).
    """
    if dt_seconds.shape != self._pending_dt.shape:
      raise ValueError(
        f"set_pending_dt expected shape {tuple(self._pending_dt.shape)}, got "
        f"{tuple(dt_seconds.shape)}"
      )
    self._pending_dt.copy_(dt_seconds.to(self._pending_dt.dtype))
    if delta_t is None:
      delta_t = dt_seconds - self._fixed_dt
    self.last_dt.copy_(delta_t.to(self.last_dt.dtype))

  def reward_timeline_scale(self) -> torch.Tensor:
    """Per-env factor ``pending_dt / fixed_dt`` for reward shaping with dynamic ``dt``.

    When ``dynamic_dt_enabled`` and the policy picks a smaller ``pending_dt`` than
    ``fixed_dt``, multiply per-step rewards by this factor so episode totals stay
    comparable in **motion-time** units (fewer seconds of clip advanced per env step).
    """
    return self._pending_dt / max(float(self._fixed_dt), 1e-8)

  def swap_reference_motion_file(self, motion_file: str) -> int:
    """Replace the loaded reference clip with another single ``.npz`` file.

    Intended for batch export tools that reuse one environment and policy:
    updates :attr:`cfg` motion fields, rebuilds :attr:`motion`, resets timeline
    bins, and clears cached debug geometry. Returns clip length ``T`` in frames.
    """
    rp = Path(motion_file).expanduser().resolve()
    if not rp.is_file():
      raise FileNotFoundError(f"Motion file not found: {rp}")
    self.cfg.motion_file = str(rp)
    self.cfg.motion_directory = ""
    self.cfg.motion_files = ()
    paths = _resolve_motion_paths(self.cfg)
    if len(paths) != 1:
      raise ValueError(f"Expected exactly one motion file, got {len(paths)}: {paths!r}")
    self.motion = MultiMotionLoader(
      paths,
      self.body_indexes,
      device=self.device,
      target_body_names=tuple(self.cfg.body_names),
      target_joint_names=G1_REPLAY_JOINT_NAMES_27,
      align_heading_to_frame=self.cfg.align_heading_to_frame,
    )
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps_progress = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    if self.motion.num_motions > 1:
      self.motion_ids[:] = torch.randint(
        0, self.motion.num_motions, (self.num_envs,), device=self.device
      )
      cl = self.motion.lengths[self.motion_ids].float()
      self.time_steps[:] = torch.minimum(
        (torch.rand(self.num_envs, device=self.device) * cl).long(),
        self.motion.lengths[self.motion_ids] - 1,
      )
      self.time_steps_progress[:] = self.time_steps
    self._sync_time_seconds_from_steps()
    self._bin_counts = torch.tensor(
      [
        max(int(m.time_step_total // (1 / self._env.step_dt)) + 1, 1)
        for m in self.motion.motions
      ],
      dtype=torch.long,
      device=self.device,
    )
    self._max_bin_count = int(self._bin_counts.max().item())
    self.bin_failed_count = torch.zeros(
      self.motion.num_motions, self._max_bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros_like(self.bin_failed_count)
    self.bin_count = self._max_bin_count
    self._ghost_model = None
    return int(self.motion.max_time_steps)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos], dim=1)

  def _gather_attr(self, attr: str) -> torch.Tensor:
    """Pick integer-index gather vs. float-time interp gather based on cfg."""
    if self.cfg.dynamic_dt_enabled:
      return self.motion._gather_interp(attr, self.motion_ids, self._time_seconds_progress)
    return self.motion._gather(attr, self.motion_ids, self.time_steps)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._gather_attr("joint_pos")

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._gather_attr("joint_vel")

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._gather_attr("body_pos_w") + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._gather_attr("body_quat_w")

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._gather_attr("body_lin_vel_w")

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._gather_attr("body_ang_vel_w")

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._gather_attr("body_pos_w")[:, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._gather_attr("body_quat_w")[:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._gather_attr("body_lin_vel_w")[:, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._gather_attr("body_ang_vel_w")[:, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    # Pick the robot joints that correspond to the motion's tracked joints
    # (handles extra robot joints that are not present in the motion clip).
    robot_subset_pos = self.robot_joint_pos[:, self._motion_to_robot_joint_ids]
    robot_subset_vel = self.robot_joint_vel[:, self._motion_to_robot_joint_ids]
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - robot_subset_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - robot_subset_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    n = len(env_ids)
    if n == 0:
      return

    mids = self.motion_ids[env_ids]
    ts = self.time_steps[env_ids]
    episode_failed = self._env.termination_manager.terminated[env_ids]

    if torch.any(episode_failed):
      mids_f = mids[episode_failed]
      ts_f = ts[episode_failed]
      bcm_f = self._bin_counts[mids_f]
      ttot_f = torch.clamp(self.motion.time_step_totals[mids_f], min=1)
      raw_bin = ts_f * bcm_f // ttot_f
      current_bin = torch.minimum(
        torch.maximum(raw_bin, torch.zeros_like(raw_bin)), bcm_f - 1
      )
      self._current_bin_failed.index_put_(
        (mids_f, current_bin),
        torch.ones(
          mids_f.shape[0],
          device=self.device,
          dtype=self._current_bin_failed.dtype,
        ),
        accumulate=True,
      )

    new_mids = torch.randint(0, self.motion.num_motions, (n,), device=self.device)
    self.motion_ids[env_ids] = new_mids

    u_ratio = self.cfg.adaptive_uniform_ratio
    M = self.motion.num_motions
    Bmax = self._max_bin_count
    bcm = self._bin_counts
    counts = self.bin_failed_count
    ksz = self.cfg.adaptive_kernel_size

    sp0 = counts + (u_ratio / bcm.float()).unsqueeze(1)
    cols = torch.arange(Bmax, device=self.device, dtype=torch.long).unsqueeze(0).expand(M, -1)
    valid_bins = cols < bcm.unsqueeze(1)
    sp0 = sp0.masked_fill(~valid_bins, 0.0)
    last_col = (bcm - 1).clamp(min=0).unsqueeze(1)
    last_vals = sp0.gather(1, last_col).expand_as(sp0)
    tail = ~valid_bins
    sp0 = torch.where(tail, last_vals, sp0)

    x = sp0.unsqueeze(1)
    padded = F.pad(x, (0, ksz - 1), mode="replicate")
    padded_b = padded.squeeze(1).unsqueeze(0)
    w = self.kernel.to(dtype=sp0.dtype, device=self.device).view(1, 1, -1).expand(M, 1, -1)
    sp_conv = F.conv1d(padded_b, w, groups=M).squeeze(0)
    sp_all = sp_conv.masked_fill(~valid_bins, 0.0)
    sp_all = sp_all / sp_all.sum(dim=1, keepdim=True).clamp(min=1e-12)

    P = sp_all[new_mids]
    bcm_n = bcm[new_mids].unsqueeze(1)
    valid_n = (
      torch.arange(Bmax, device=self.device, dtype=torch.long).unsqueeze(0).expand(n, -1)
      < bcm_n
    )
    P = P.masked_fill(~valid_n, 0.0)
    P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-12)
    sampled_bins = torch.multinomial(P, num_samples=1, replacement=True).squeeze(1)

    bcm_nf = bcm[new_mids].float()
    ttot_mf = torch.clamp(self.motion.time_step_totals[new_mids] - 1, min=0).float()
    u_rand = sample_uniform(0.0, 1.0, (n,), device=self.device)
    new_ts = ((sampled_bins.float() + u_rand) / bcm_nf * ttot_mf).long()

    self.time_steps[env_ids] = torch.minimum(
      new_ts,
      torch.clamp(self.motion.time_step_totals[new_mids] - 1, min=0),
    )
    self.time_steps_progress[env_ids] = self.time_steps[env_ids]
    self._sync_time_seconds_from_steps(env_ids)

    H = -(sp_all * (sp_all + 1e-12).log()).sum(dim=1)
    bcm_f = bcm.float()
    H_norm = torch.where(bcm > 1, H / torch.log(bcm_f), torch.ones_like(H))
    pmax, imax = sp_all.max(dim=1)
    imax_norm = imax.float() / bcm_f.clamp(min=1)
    sel_counts = torch.bincount(new_mids, minlength=M)
    mask_m = sel_counts > 0
    if mask_m.any():
      self.metrics["sampling_entropy"][:] = H_norm[mask_m].mean()
      self.metrics["sampling_top1_prob"][:] = pmax[mask_m].float().mean()
      self.metrics["sampling_top1_bin"][:] = imax_norm[mask_m].mean()

  def _uniform_sampling(self, env_ids: torch.Tensor):
    n = len(env_ids)
    if n == 0:
      return
    mids = torch.randint(0, self.motion.num_motions, (n,), device=self.device)
    self.motion_ids[env_ids] = mids
    T = self.motion.time_step_totals[mids]
    r = sample_uniform(0.0, 1.0, (n,), device=self.device)
    self.time_steps[env_ids] = torch.minimum(
      (r * T.float()).long(),
      torch.clamp(T - 1, min=0),
    )
    self.time_steps_progress[env_ids] = self.time_steps[env_ids]
    self._sync_time_seconds_from_steps(env_ids)
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform time draw.
    mean_inv_bins = float((1.0 / self._bin_counts.float()).mean().item())
    self.metrics["sampling_top1_prob"][:] = mean_inv_bins
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    # Motion clips may contain a subset of robot joints (e.g., 27-dof motion on a
    # robot with extra untracked joints). Map each motion column
    # to the matching robot joint slot via ``_motion_to_robot_joint_ids`` instead
    # of assuming motion col k == robot col k.
    n_motion_joints = joint_pos.shape[1]
    expected_motion_joints = int(self._motion_to_robot_joint_ids.numel())
    if n_motion_joints != expected_motion_joints:
      raise ValueError(
        f"Motion joint dim ({n_motion_joints}) does not match the mapping "
        f"length ({expected_motion_joints})."
      )
    joint_ids = self._motion_to_robot_joint_ids
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(
      joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids
    )

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      if self.motion.num_motions > 1:
        self.motion_ids[env_ids] = torch.randint(
          0, self.motion.num_motions, (len(env_ids),), device=self.device
        )
      self.time_steps[env_ids] = 0
      self.time_steps_progress[env_ids] = 0
      self._sync_time_seconds_from_steps(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    warmup_mask = self._motion_warmup_mask()[env_ids]
    if torch.any(warmup_mask):
      warm_ids = env_ids[warmup_mask]
      self._freeze_motion_timeline_at_start(warm_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    self.root_pos_rand_offset_w[env_ids] = rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    self.root_ori_rand_offset_w[env_ids] = orientations_delta
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self):
    clip_lens = self.motion.lengths[self.motion_ids]
    last_idx = torch.clamp(clip_lens - 1, min=0)
    #print(f"motion_time:{self._time_seconds_progress}, last_dt:{self.last_dt}")
    if self.cfg.dynamic_dt_enabled:
      if self.cfg.random_dt_training_enabled:
        lo, hi = float(self.cfg.dt_delta_range[0]), float(self.cfg.dt_delta_range[1])
        if lo > hi:
          lo, hi = hi, lo
        delta = sample_uniform(lo, hi, (self.num_envs,), device=self.device)
        dt_tensor = torch.full(
          (self.num_envs,), float(self._fixed_dt), dtype=torch.float32, device=self.device
        ) + delta.to(dtype=torch.float32)
        self.set_pending_dt(dt_tensor, delta_t=delta.to(dtype=torch.float32))
      # Advance per-env float timeline by the most recent ``dt`` sampled above.
      # ``time_steps_progress`` mirrors the rounded integer for legacy consumers
      # (termination, debug viz, adaptive bins); the canonical state is the float timeline.
      fps = torch.clamp(self.motion.fps_values[self.motion_ids], min=1.0)
      self._time_seconds_progress = (self._time_seconds_progress + self._pending_dt).clamp_min(0.0)
      # Reset pending dt to baseline so a missing action term (or a one-step
      # action gap) still advances at the fixed cadence next step.
      self._pending_dt[:] = self._fixed_dt
      step_f = torch.round(self._time_seconds_progress * fps).to(torch.long).clamp_min(0)
      self.time_steps_progress = step_f
      self.time_steps = torch.minimum(self.time_steps_progress, last_idx)
    else:
      self.time_steps_progress += 1
      # Keep tracking the last frame after reaching clip end (e.g., during warmdown),
      # instead of re-sampling from the beginning.
      self.time_steps = torch.minimum(self.time_steps_progress, last_idx)
      # Keep float timeline in sync so ``current_time_seconds`` remains a single
      # source of truth even when downstream code mixes modes.
      fps = torch.clamp(self.motion.fps_values[self.motion_ids], min=1.0)
      self._time_seconds_progress = self.time_steps.to(torch.float32) / fps

    warmup_mask = self._motion_warmup_mask()
    if torch.any(warmup_mask):
      warm_ids = warmup_mask.nonzero(as_tuple=False).flatten()
      self._freeze_motion_timeline_at_start(warm_ids)

    self.update_relative_body_poses()

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()
    if self._dump_txt_path:
      obs_like: torch.Tensor | None = None
      if isinstance(getattr(self._env, "obs_buf", None), dict):
        actor = self._env.obs_buf.get("actor")  # type: ignore[union-attr]
        if isinstance(actor, torch.Tensor):
          obs_like = actor.detach().cpu()
      if obs_like is None:
        # Fallback: command vector (e.g., very early startup before obs_buf is ready).
        obs_like = self.command.detach().cpu()
      with open(self._dump_txt_path, "a", encoding="utf-8") as f:
        for env_id in range(obs_like.shape[0]):
          vals = " ".join(f"{float(v):.6f}" for v in obs_like[env_id])
          f.write(f"{self._dump_step} {env_id} {vals}\n")
      self._dump_step += 1

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      motion_to_robot_np = self._motion_to_robot_joint_ids.cpu().numpy()
      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        joint_pos = self.joint_pos[batch].cpu().numpy()
        # Place each motion joint value at the qpos address of the matching robot
        # joint, leaving untracked robot joints at 0.
        qpos[joint_q_adr[motion_to_robot_np]] = joint_pos

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.max_time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        mid = int(self.motion_ids[idx].item())
        t_hi = int(self.motion.motions[mid].time_step_total) - 1
        self.time_steps[idx] = min(int(scrubber.value), max(t_hi, 0))
        self.time_steps_progress[idx] = self.time_steps[idx]
        idx_t = torch.tensor([idx], dtype=torch.long, device=self.device)
        self._sync_time_seconds_from_steps(idx_t)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    """Reset to exact reference state at a specific frame.

    Like ``_resample_command`` but deterministic: no random
    perturbations to pose, velocity, or joint positions.
    """
    max_t = self.motion.time_step_totals[self.motion_ids[env_ids]] - 1
    self.time_steps[env_ids] = torch.minimum(
      torch.full_like(max_t, frame, dtype=torch.long),
      torch.clamp(max_t, min=0),
    )
    self.time_steps_progress[env_ids] = self.time_steps[env_ids]
    self._sync_time_seconds_from_steps(env_ids)
    self.root_pos_rand_offset_w[env_ids] = 0.0
    self.root_ori_rand_offset_w[env_ids] = 0.0
    self.root_ori_rand_offset_w[env_ids, 0] = 1.0
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )

  def current_time_seconds(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Return per-env motion timeline time in seconds.

    When ``dynamic_dt_enabled``, returns the float timeline driven by
    :attr:`_pending_dt`. Otherwise, derives it from the integer ``time_steps``
    (legacy behavior).
    """
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
    if self.cfg.dynamic_dt_enabled:
      t = self._time_seconds_progress[env_ids]
      # Clamp to clip duration so callers do not run past the last frame.
      fps = torch.clamp(self.motion.fps_values[self.motion_ids[env_ids]], min=1.0)
      max_t = (self.motion.time_step_totals[self.motion_ids[env_ids]] - 1).clamp(min=0).to(t.dtype) / fps
      #print(t)
      return torch.minimum(torch.clamp_min(t, 0.0), max_t)
    return self.motion.gather_time_seconds(self.motion_ids[env_ids], self.time_steps[env_ids])


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_files: tuple[str, ...] = ()
  """Explicit list of ``.npz`` clips (highest priority)."""

  motion_directory: str = ""
  """Directory of ``*.npz`` (sorted); used when ``motion_files`` is empty."""

  motion_file: str = ""
  """Single ``.npz`` path when ``motion_files`` and ``motion_directory`` are empty."""

  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  align_heading_to_frame: AlignHeadingFrame = "last"
  """World-yaw alignment for loaded clips (AdaPT Tennis).

  - ``last`` (default): rotate the clip so the **last** frame root faces +X.
    Matches serve stance (sideways at start, facing court after the swing).
  - ``first``: legacy behavior — first-frame root yaw -> 0.
  - ``none``: leave npz orientations unchanged.
  """
  dump_obs_txt_path: str = ""
  """If non-empty, append each step's ``command`` vector to this txt file."""

  # --- Dynamic motion-time-step (dt) options ---------------------------------
  # When ``dynamic_dt_enabled`` is True, the per-step advance of the motion
  # timeline is no longer a fixed integer frame but a per-env float
  # ``dt`` in seconds (env-sampled in stage-1 random-dt training). The
  # reference frame is then drawn by linear interpolation in time (slerp for
  # quaternions) instead of nearest-integer indexing.
  dynamic_dt_enabled: bool = False
  """Enable per-env policy-controlled motion ``dt`` advance + interpolation."""

  fixed_dt: float = 0.02
  """Baseline ``dt`` (seconds) when no per-step override is provided.
  Final ``dt`` is ``fixed_dt + delta_t``."""

  dt_delta_range: tuple[float, float] = (-0.01, 0.02)
  """Clamp range for the policy-output residual ``delta_t`` (seconds). Default
  permits ``dt`` in ``[0.01, 0.04]`` around the 0.02 baseline (0.5x ~ 2x)."""

  dt_dead_band_threshold: float = 0.0
  """Magnitude threshold below which ``delta_t`` is forced to zero. ``0.0``
  disables the dead-band."""

  random_dt_training_enabled: bool = False
  """If True with ``dynamic_dt_enabled``, sample ``delta_t`` uniformly from
  ``dt_delta_range`` inside :meth:`MotionCommand._update_command` each control
  step (no policy ``motion_dt`` action). Intended for stage-1 curriculum: the
  policy only outputs joint actions while observations expose the sampled
  ``delta_t`` (see :func:`mjlab.tasks.adapt_tennis.mdp.observations.motion_dt_sample`).
  Mutually exclusive with adding a policy-driven ``motion_dt`` action term."""

  motion_warmup_steps: int = 0
  """While the training step counter (see ``motion_warmup_step_source``) is below
  this value, the motion reference stays at frame 0 so the robot can stabilize
  before timeline tracking begins. ``0`` disables warmup."""

  motion_warmup_step_source: Literal["env", "learning_iteration", "episode"] = "env"
  """How to count warmup steps (see :attr:`motion_warmup_steps`):

  - ``env``: global :attr:`~mjlab.envs.ManagerBasedRlEnv.common_step_counter``
    (works in train; in play, counter starts at 0 because actor-only checkpoint
    load does not restore the training counter)
  - ``learning_iteration``: PPO iteration on the env (training)
  - ``episode``: per-env :attr:`~mjlab.envs.ManagerBasedRlEnv.episode_length_buf``
    (recommended for **play** / viewer reset; re-warms every episode)"""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)

