from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def motion_dt_prev(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Residual ``delta_t`` (seconds) associated with the motion-timeline advance.

  Returns ``[num_envs, 1]``. With ``MotionCommandCfg.random_dt_training_enabled``,
  this is the ``delta_t`` sampled at the most recent
  :meth:`MotionCommand._update_command` (same source as :func:`motion_dt_sample`).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.last_dt.unsqueeze(-1)


def motion_dt_sample(
  env: ManagerBasedRlEnv, command_name: str = "motion"
) -> torch.Tensor:
  """Alias of :func:`motion_dt_prev` for stage-1 random-``dt`` curriculum (1-dim)."""
  return motion_dt_prev(env, command_name=command_name)


def motion_phase(env: ManagerBasedRlEnv, command_name: str = "motion") -> torch.Tensor:
  """Normalized motion phase in [0, 1] for current reference clip time."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  motion_ids = command.motion_ids[env_ids]
  clip_steps = torch.clamp(
    command.motion.time_step_totals[motion_ids] - 1, min=1
  ).to(torch.float32)
  clip_fps = torch.clamp(command.motion.fps_values[motion_ids], min=1.0)
  clip_total_t = clip_steps / clip_fps
  if command.cfg.dynamic_dt_enabled:
    t_now = command.current_time_seconds(env_ids)
  else:
    t_now = command.time_steps[env_ids].to(torch.float32) / clip_fps
  phase = torch.clamp(t_now / torch.clamp(clip_total_t, min=1.0e-6), 0.0, 1.0)
  return phase.unsqueeze(-1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
