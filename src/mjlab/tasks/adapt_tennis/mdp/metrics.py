from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommand


def compute_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Mean Per-Keybody Position Error (MPKPE).

  MPKPE measures the average Euclidean distance between the reference and
  actual positions of all key bodies in world frame.
  """
  pos_error = command.body_pos_relative_w - command.robot_body_pos_w
  per_body_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_bodies)
  return per_body_error.mean(dim=-1)  # (num_envs,)


def _reference_body_pos_yaw_aligned_w(command: MotionCommand) -> torch.Tensor:
  """Reference keybody positions with anchor yaw/height alignment.

  Matches :meth:`MotionCommand.update_relative_body_poses` (used by ``body_pos_relative_w``):

  - Place the reference at the robot anchor XY and reference anchor Z.
  - Rotate reference offsets by the yaw delta between robot and reference anchors.
  """
  num_bodies = len(command.cfg.body_names)
  anchor_pos = command.anchor_pos_w.unsqueeze(1).expand(-1, num_bodies, 3)
  anchor_quat = command.anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, 4)
  robot_anchor_pos = command.robot_anchor_pos_w.unsqueeze(1).expand(-1, num_bodies, 3)
  robot_anchor_quat = command.robot_anchor_quat_w.unsqueeze(1).expand(-1, num_bodies, 4)

  delta_pos = robot_anchor_pos.clone()
  delta_pos[..., 2] = anchor_pos[..., 2]
  delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
  return delta_pos + quat_apply(delta_ori, command.body_pos_w - anchor_pos)


def compute_yaw_aligned_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Root-relative MPKPE with global anchor yaw removed (R-MPKPE + yaw alignment).

  Same convention as :func:`compute_mpkpe` / ``Metrics/motion/error_body_pos``: compares
  yaw-aligned reference bodies to robot world positions. Invariant to global yaw about
  vertical axis; still sensitive to anchor roll/pitch mismatch and vertical offset handling.
  """
  ref_aligned = _reference_body_pos_yaw_aligned_w(command)
  per_body_error = torch.norm(ref_aligned - command.robot_body_pos_w, dim=-1)
  return per_body_error.mean(dim=-1)


def compute_root_relative_mpkpe(command: MotionCommand) -> torch.Tensor:
  """Compute Root-relative Mean Per-Keybody Position Error (R-MPKPE).

  R-MPKPE subtracts each side's anchor position only (no orientation normalization).
  For yaw-aligned error use :func:`compute_yaw_aligned_root_relative_mpkpe`.
  """
  ref_anchor_pos = command.anchor_pos_w.unsqueeze(1)
  ref_rel_pos = command.body_pos_w - ref_anchor_pos

  robot_anchor_pos = command.robot_anchor_pos_w.unsqueeze(1)
  robot_rel_pos = command.robot_body_pos_w - robot_anchor_pos

  pos_error = ref_rel_pos - robot_rel_pos
  per_body_error = torch.norm(pos_error, dim=-1)
  return per_body_error.mean(dim=-1)


def compute_joint_position_error(command: MotionCommand) -> torch.Tensor:
  """L2 norm of joint position error over motion replay joints (same as ``error_joint_pos``)."""
  robot_subset_pos = command.robot_joint_pos[:, command._motion_to_robot_joint_ids]
  return torch.norm(command.joint_pos - robot_subset_pos, dim=-1)


def compute_joint_velocity_error(command: MotionCommand) -> torch.Tensor:
  """Joint velocity error (rad/s L2 norm over motion replay joints)."""
  robot_subset_vel = command.robot_joint_vel[:, command._motion_to_robot_joint_ids]
  return torch.norm(command.joint_vel - robot_subset_vel, dim=-1)


def compute_mean_joint_acceleration(command: MotionCommand) -> torch.Tensor:
  """Per-env mean absolute joint acceleration (lower is smoother)."""
  acc = command.robot.data.joint_acc
  return acc.abs().mean(dim=-1)


def compute_ee_position_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector position error."""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_pos = command.body_pos_relative_w[:, ee_indices]
  robot_ee_pos = command.robot_body_pos_w[:, ee_indices]

  pos_error = ref_ee_pos - robot_ee_pos
  per_ee_error = torch.norm(pos_error, dim=-1)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def compute_ee_orientation_error(
  command: MotionCommand,
  ee_body_names: tuple[str, ...],
) -> torch.Tensor:
  """Compute end effector orientation error."""
  ee_indices = _get_body_indices(command, ee_body_names)
  if len(ee_indices) == 0:
    return torch.zeros(command.num_envs, device=command.device)

  ref_ee_quat = command.body_quat_relative_w[:, ee_indices]
  robot_ee_quat = command.robot_body_quat_w[:, ee_indices]

  per_ee_error = quat_error_magnitude(ref_ee_quat, robot_ee_quat)  # (num_envs, num_ee)
  return per_ee_error.mean(dim=-1)  # (num_envs,)


def _get_body_indices(
  command: MotionCommand,
  body_names: tuple[str, ...],
) -> list[int]:
  """Get indices of specified bodies within the command's body list.

  Args:
    command: The motion command.
    body_names: Names of bodies to find.

  Returns:
    List of indices into command.cfg.body_names.
  """
  return [i for i, name in enumerate(command.cfg.body_names) if name in body_names]
