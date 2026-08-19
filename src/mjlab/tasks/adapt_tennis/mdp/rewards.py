from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from mjlab.tasks.adapt_tennis.replay_joint_names import G1_REPLAY_JOINT_NAMES_27

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def _get_motion_joint_indexes(joint_names: tuple[str, ...]) -> list[int]:
  """Indices into motion / robot replay joint vectors (``G1_REPLAY_JOINT_NAMES_27``)."""
  missing = [n for n in joint_names if n not in G1_REPLAY_JOINT_NAMES_27]
  if missing:
    raise ValueError(
      f"Unknown joint name(s) for motion replay tracking: {missing}. "
      f"Expected names from G1_REPLAY_JOINT_NAMES_27."
    )
  return [G1_REPLAY_JOINT_NAMES_27.index(n) for n in joint_names]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  std: float = 0.2,
  joint_names: tuple[str, ...] = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
  ),
) -> torch.Tensor:
  """Reward tracking selected motion-reference joint positions (``dof_pos``).

  Compares ``MotionCommand.joint_pos`` to the robot subset aligned with the motion
  replay layout (same indexing as ``error_joint_pos`` metrics).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_indexes = _get_motion_joint_indexes(joint_names)
  if not joint_indexes:
    return torch.zeros(command.num_envs, dtype=torch.float32, device=command.device)

  motion_j = command.joint_pos[:, joint_indexes]
  robot_j = command.robot_joint_pos[:, command._motion_to_robot_joint_ids][
    :, joint_indexes
  ]
  error_sq = torch.sum((motion_j - robot_j) ** 2, dim=-1)
  return torch.exp(-error_sq / max(float(std) ** 2, 1.0e-12))


class motion_keyframe_joint_position_reward:
  """Sparse reward for matching robot ``dof_pos`` near motion keyframe times.

  At each configured keyframe time ``t_k`` (seconds on the motion clip), when
  ``|current_time - t_k| <= window_s`` the reward compares the robot's selected
  joints to the motion reference ``joint_pos`` interpolated at ``t_k``.
  """

  def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv"):
    params = cfg.params
    command_name = params.get("command_name", "motion")
    joint_names = params["joint_names"]
    keyframe_times = params["keyframe_times"]
    if isinstance(joint_names, str):
      joint_names = (joint_names,)
    if isinstance(keyframe_times, (int, float)):
      keyframe_times = (float(keyframe_times),)
    keyframe_times = tuple(float(t) for t in keyframe_times)
    if not keyframe_times:
      raise ValueError("motion_keyframe_joint_position_reward requires keyframe_times.")

    self._command_name = command_name
    self._joint_indexes = _get_motion_joint_indexes(tuple(joint_names))
    self._window_s = float(params.get("window_s", 0.05))
    self._std = float(params.get("std", 0.15))

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    self._motion_to_robot_joint_ids = command._motion_to_robot_joint_ids
    motion = command.motion
    num_motions = motion.num_motions
    device = env.device
    kf_targets = []
    for t_s in keyframe_times:
      t = torch.full((num_motions,), t_s, dtype=torch.float32, device=device)
      mids = torch.arange(num_motions, device=device, dtype=torch.long)
      joint_pos = motion._gather_interp("joint_pos", mids, t)
      kf_targets.append(joint_pos[:, self._joint_indexes])
    self._keyframe_targets = torch.stack(kf_targets, dim=1)
    self._keyframe_times = torch.tensor(
      keyframe_times, dtype=torch.float32, device=device
    )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str = "motion",
    keyframe_times: tuple[float, ...] = (),
    joint_names: tuple[str, ...] = (),
    std: float = 0.15,
    window_s: float = 0.05,
  ) -> torch.Tensor:
    del command_name, keyframe_times, joint_names, std, window_s

    command = cast(MotionCommand, env.command_manager.get_term(self._command_name))
    t_now = command.current_time_seconds()
    motion_ids = command.motion_ids
    robot_j = command.robot_joint_pos[:, self._motion_to_robot_joint_ids][
      :, self._joint_indexes
    ]

    time_diff = (t_now.unsqueeze(-1) - self._keyframe_times.unsqueeze(0)).abs()
    in_window = time_diff <= self._window_s
    if not torch.any(in_window):
      return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    targets = self._keyframe_targets[motion_ids]
    error_sq = torch.sum((targets - robot_j.unsqueeze(1)) ** 2, dim=-1)
    kf_rew = torch.exp(-error_sq / max(self._std**2, 1.0e-12))
    kf_rew = kf_rew * in_window.to(kf_rew.dtype)
    return kf_rew.max(dim=-1).values


class motion_keyframe_relative_body_position_reward:
  """Sparse reward for matching yaw-aligned relative body positions at keyframes.

  ``keyframe_times`` may contain one or many times. At each ``t_k``, when
  ``|current_time - t_k| <= window_s`` the term compares selected bodies to the
  motion pose at ``t_k`` (same residual as ``motion_relative_body_position``).
  Active keyframe rewards are **summed** (so multiple keyframes all contribute).
  """

  def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv"):
    params = cfg.params
    command_name = params.get("command_name", "motion")
    body_names = params["body_names"]
    keyframe_times = params["keyframe_times"]
    if isinstance(body_names, str):
      body_names = (body_names,)
    if isinstance(keyframe_times, (int, float)):
      keyframe_times = (float(keyframe_times),)
    keyframe_times = tuple(float(t) for t in keyframe_times)
    if not keyframe_times:
      raise ValueError(
        "motion_keyframe_relative_body_position_reward requires keyframe_times."
      )
    if not body_names:
      raise ValueError(
        "motion_keyframe_relative_body_position_reward requires body_names."
      )

    self._command_name = command_name
    self._window_s = float(params.get("window_s", 0.05))
    self._std = float(params.get("std", 0.15))

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    self._body_indexes = _get_body_indexes(command, tuple(body_names))
    if not self._body_indexes:
      raise ValueError(
        f"None of body_names={body_names} found in motion cfg.body_names="
        f"{command.cfg.body_names}."
      )
    self._anchor_body_index = command.motion_anchor_body_index
    self._num_tracked_bodies = len(command.cfg.body_names)

    motion = command.motion
    num_motions = motion.num_motions
    device = env.device
    kf_body_pos = []
    kf_body_quat = []
    for t_s in keyframe_times:
      t = torch.full((num_motions,), t_s, dtype=torch.float32, device=device)
      mids = torch.arange(num_motions, device=device, dtype=torch.long)
      kf_body_pos.append(motion._gather_interp("body_pos_w", mids, t))
      kf_body_quat.append(motion._gather_interp("body_quat_w", mids, t))
    self._keyframe_body_pos = torch.stack(kf_body_pos, dim=1)
    self._keyframe_body_quat = torch.stack(kf_body_quat, dim=1)
    self._keyframe_times = torch.tensor(
      keyframe_times, dtype=torch.float32, device=device
    )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str = "motion",
    keyframe_times: tuple[float, ...] = (),
    body_names: tuple[str, ...] = (),
    std: float = 0.15,
    window_s: float = 0.05,
  ) -> torch.Tensor:
    del command_name, keyframe_times, body_names, std, window_s

    command = cast(MotionCommand, env.command_manager.get_term(self._command_name))
    t_now = command.current_time_seconds()
    motion_ids = command.motion_ids

    time_diff = (t_now.unsqueeze(-1) - self._keyframe_times.unsqueeze(0)).abs()
    in_window = time_diff <= self._window_s
    if not torch.any(in_window):
      return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    origins = env.scene.env_origins[:, None, None, :]
    body_pos_kf = self._keyframe_body_pos[motion_ids] + origins
    body_quat_kf = self._keyframe_body_quat[motion_ids]
    anchor_pos_kf = body_pos_kf[:, :, self._anchor_body_index, :]
    anchor_quat_kf = body_quat_kf[:, :, self._anchor_body_index, :]

    n_bodies = self._num_tracked_bodies
    robot_anchor_pos = command.robot_anchor_pos_w[:, None, None, :].expand(
      -1, body_pos_kf.shape[1], n_bodies, -1
    )
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, None, :].expand(
      -1, body_pos_kf.shape[1], n_bodies, -1
    )
    anchor_pos = anchor_pos_kf[:, :, None, :].expand(-1, -1, n_bodies, -1)
    anchor_quat = anchor_quat_kf[:, :, None, :].expand(-1, -1, n_bodies, -1)

    delta_pos = robot_anchor_pos.clone()
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
    body_pos_relative = delta_pos + quat_apply(delta_ori, body_pos_kf - anchor_pos)

    robot_body = command.robot_body_pos_w[:, self._body_indexes]
    ref_body = body_pos_relative[:, :, self._body_indexes, :]
    error_per_body = torch.sum(
      torch.square(ref_body - robot_body.unsqueeze(1)), dim=-1
    )
    error = error_per_body.mean(dim=-1)
    kf_rew = torch.exp(-error / max(self._std**2, 1.0e-12))
    kf_rew = kf_rew * in_window.to(kf_rew.dtype)
    return kf_rew.sum(dim=-1)


class motion_keyframe_racket_site_position_reward:
  """Sparse reward for matching yaw-aligned ``racket_point`` site position at keyframes."""

  def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv"):
    params = cfg.params
    command_name = params.get("command_name", "motion")
    site_name = params.get("site_name", "racket_point")
    site_ref_body_name = params["site_ref_body_name"]
    robot_entity_name = params.get("robot_entity_name", "robot")
    keyframe_times = params["keyframe_times"]
    if isinstance(keyframe_times, (int, float)):
      keyframe_times = (float(keyframe_times),)
    keyframe_times = tuple(float(t) for t in keyframe_times)
    if not keyframe_times:
      raise ValueError(
        "motion_keyframe_racket_site_position_reward requires keyframe_times."
      )

    self._command_name = command_name
    self._window_s = float(params.get("window_s", 0.05))
    self._std = float(params.get("std", 0.15))

    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    if site_ref_body_name not in command.cfg.body_names:
      raise ValueError(
        f"site_ref_body_name={site_ref_body_name!r} not in motion cfg.body_names="
        f"{command.cfg.body_names}."
      )
    self._ref_body_index = command.cfg.body_names.index(site_ref_body_name)
    self._anchor_body_index = command.motion_anchor_body_index

    robot = env.scene[robot_entity_name]
    if site_name not in robot.site_names:
      raise ValueError(
        f"site_name={site_name!r} not found in robot sites {robot.site_names}."
      )
    if site_ref_body_name not in robot.body_names:
      raise ValueError(
        f"site_ref_body_name={site_ref_body_name!r} not found in robot bodies."
      )
    self._site_index = robot.site_names.index(site_name)
    self._robot_ref_body_index = robot.body_names.index(site_ref_body_name)
    self._robot_entity_name = robot_entity_name
    self._site_local_offset: torch.Tensor | None = None

    motion = command.motion
    num_motions = motion.num_motions
    device = env.device
    kf_body_pos = []
    kf_body_quat = []
    for t_s in keyframe_times:
      t = torch.full((num_motions,), t_s, dtype=torch.float32, device=device)
      mids = torch.arange(num_motions, device=device, dtype=torch.long)
      kf_body_pos.append(motion._gather_interp("body_pos_w", mids, t))
      kf_body_quat.append(motion._gather_interp("body_quat_w", mids, t))
    self._keyframe_body_pos = torch.stack(kf_body_pos, dim=1)
    self._keyframe_body_quat = torch.stack(kf_body_quat, dim=1)
    self._keyframe_times = torch.tensor(
      keyframe_times, dtype=torch.float32, device=device
    )

  def _ensure_site_local_offset(self, env: "ManagerBasedRlEnv") -> torch.Tensor:
    if self._site_local_offset is not None:
      return self._site_local_offset
    robot = env.scene[self._robot_entity_name]
    site_pos_w = robot.data.site_pos_w[0, self._site_index]
    ref_pos_w = robot.data.body_link_pos_w[0, self._robot_ref_body_index]
    ref_quat_w = robot.data.body_link_quat_w[0, self._robot_ref_body_index]
    self._site_local_offset = quat_apply_inverse(
      ref_quat_w, site_pos_w - ref_pos_w
    ).detach()
    return self._site_local_offset

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str = "motion",
    keyframe_times: tuple[float, ...] = (),
    site_name: str = "racket_point",
    site_ref_body_name: str = "",
    robot_entity_name: str = "robot",
    std: float = 0.15,
    window_s: float = 0.05,
  ) -> torch.Tensor:
    del command_name, keyframe_times, site_name, site_ref_body_name
    del robot_entity_name, std, window_s

    command = cast(MotionCommand, env.command_manager.get_term(self._command_name))
    t_now = command.current_time_seconds()
    motion_ids = command.motion_ids

    time_diff = (t_now.unsqueeze(-1) - self._keyframe_times.unsqueeze(0)).abs()
    in_window = time_diff <= self._window_s
    if not torch.any(in_window):
      return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    site_local_offset = self._ensure_site_local_offset(env)

    origins = env.scene.env_origins[:, None, None, :]
    body_pos_kf = self._keyframe_body_pos[motion_ids] + origins
    body_quat_kf = self._keyframe_body_quat[motion_ids]
    anchor_pos_kf = body_pos_kf[:, :, self._anchor_body_index, :]
    anchor_quat_kf = body_quat_kf[:, :, self._anchor_body_index, :]

    ref_pos_kf = body_pos_kf[:, :, self._ref_body_index, :]
    ref_quat_kf = body_quat_kf[:, :, self._ref_body_index, :]
    local = site_local_offset.view(1, 1, 3).expand_as(ref_pos_kf)
    site_pos_kf = ref_pos_kf + quat_apply(ref_quat_kf, local)

    n_kf = site_pos_kf.shape[1]
    robot_anchor_pos = command.robot_anchor_pos_w[:, None, :].expand(-1, n_kf, -1)
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(-1, n_kf, -1)
    anchor_pos = anchor_pos_kf
    anchor_quat = anchor_quat_kf

    delta_pos = robot_anchor_pos.clone()
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
    site_pos_relative = delta_pos + quat_apply(delta_ori, site_pos_kf - anchor_pos)

    robot = env.scene[self._robot_entity_name]
    robot_site = robot.data.site_pos_w[:, self._site_index]
    error_sq = torch.sum(
      (site_pos_relative - robot_site.unsqueeze(1)) ** 2, dim=-1
    )
    kf_rew = torch.exp(-error_sq / max(self._std**2, 1.0e-12))
    kf_rew = kf_rew * in_window.to(kf_rew.dtype)
    return kf_rew.sum(dim=-1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    hit = (force_mag > force_threshold).any(dim=1)
    return hit.sum(dim=-1).float()
  assert data.found is not None
  return data.found.squeeze(-1)
