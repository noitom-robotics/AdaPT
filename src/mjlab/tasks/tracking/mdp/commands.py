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


class MotionLoader:
  """Loads one motion clip from ``.npz``."""

  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
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
    # self._body_indexes = body_indexes
    # import pdb; pdb.set_trace()
    # ``body_indexes`` are indices into the *simulation* robot body array (from
    # ``find_bodies``).  Some ``.npz`` files store ``body_*`` with one row per
    # *tracked* body only (same order as ``cfg.body_names``), so dim-1 length
    # equals ``len(body_indexes)`` but values like 30 would be out of range.
    # In that compact layout, use columns ``0 .. N-1`` instead.
    bi = body_indexes.to(device=device)
    n_body_motion = int(self._body_pos_w.shape[1])
    if bi.numel() and int(bi.max().item()) >= n_body_motion:
      motion_body_cols = torch.arange(bi.shape[0], dtype=torch.long, device=device)
      assert len(motion_body_cols) == len(body_indexes)
    else:
      motion_body_cols = bi
    self._body_indexes = motion_body_cols
    self.body_pos_w = self._body_pos_w[:, self._body_indexes]
    self.body_quat_w = self._body_quat_w[:, self._body_indexes]
    self.body_lin_vel_w = self._body_lin_vel_w[:, self._body_indexes]
    self.body_ang_vel_w = self._body_ang_vel_w[:, self._body_indexes]
    self.time_step_total = self.joint_pos.shape[0]


class MultiMotionLoader:
  """Holds multiple ``MotionLoader`` clips; gather by per-env ``motion_ids`` + ``time_steps``."""

  def __init__(
    self, motion_paths: tuple[str, ...], body_indexes: torch.Tensor, device: str
  ) -> None:
    if not motion_paths:
      raise ValueError("MultiMotionLoader requires at least one motion path.")
    self.motions = [
      MotionLoader(p, body_indexes, device=device) for p in motion_paths
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
    self.device = device
    self.time_step_totals = torch.tensor(
      [m.time_step_total for m in self.motions],
      dtype=torch.long,
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
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    paths = _resolve_motion_paths(cfg)
    self.motion = MultiMotionLoader(paths, self.body_indexes, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

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
    self.motion = MultiMotionLoader(paths, self.body_indexes, device=self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    if self.motion.num_motions > 1:
      self.motion_ids[:] = torch.randint(
        0, self.motion.num_motions, (self.num_envs,), device=self.device
      )
      cl = self.motion.lengths[self.motion_ids].float()
      self.time_steps[:] = torch.minimum(
        (torch.rand(self.num_envs, device=self.device) * cl).long(),
        self.motion.lengths[self.motion_ids] - 1,
      )
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

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.gather_joint_pos(self.motion_ids, self.time_steps)

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.gather_joint_vel(self.motion_ids, self.time_steps)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.gather_body_pos_w(self.motion_ids, self.time_steps)
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.gather_body_quat_w(self.motion_ids, self.time_steps)

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.gather_body_lin_vel_w(self.motion_ids, self.time_steps)

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.gather_body_ang_vel_w(self.motion_ids, self.time_steps)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.gather_body_pos_w(self.motion_ids, self.time_steps)[
        :, self.motion_anchor_body_index
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.gather_body_quat_w(self.motion_ids, self.time_steps)[
      :, self.motion_anchor_body_index
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.gather_body_lin_vel_w(self.motion_ids, self.time_steps)[
      :, self.motion_anchor_body_index
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.gather_body_ang_vel_w(self.motion_ids, self.time_steps)[
      :, self.motion_anchor_body_index
    ]

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

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
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
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
      if self.motion.num_motions > 1:
        self.motion_ids[env_ids] = torch.randint(
          0, self.motion.num_motions, (len(env_ids),), device=self.device
        )
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

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
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
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
    self.time_steps += 1
    clip_lens = self.motion.lengths[self.motion_ids]
    env_ids = torch.where(self.time_steps >= clip_lens)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

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

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

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
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


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
  dump_obs_txt_path: str = ""
  """If non-empty, append each step's ``command`` vector to this txt file."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
