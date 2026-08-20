"""Stage-1 motion-mimic env config with random motion-``dt`` curriculum.

``make_stage1_tracking_env_cfg`` mirrors ``mjlab.tasks.tracking.tracking_env_cfg``
layout (single ``actor`` / ``critic``, motion command only).
"""

from __future__ import annotations

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.adapt_tennis import mdp
from mjlab.tasks.adapt_tennis.mdp import MotionCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

# Hitting-arm key bodies. Switch ``HIT_ARM_BODY_NAMES`` to the active side.
_HIT_ARM_BODY_NAMES_LEFT: tuple[str, ...] = (
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
)
_HIT_ARM_BODY_NAMES_RIGHT: tuple[str, ...] = (
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
HIT_ARM_BODY_NAMES: tuple[str, ...] = _HIT_ARM_BODY_NAMES_LEFT

# Motion-clip keyframe times (seconds) for sparse hit-arm pose shaping.
HIT_ARM_KEYFRAME_TIMES_S: tuple[float, ...] = (3.4,)
# player1: 3.4
# player2: 1.84

def _hit_arm_side(
  racket_hand: Literal["left", "right"] | None = None,
) -> Literal["left", "right"]:
  if racket_hand is not None:
    return racket_hand
  return "left" if HIT_ARM_BODY_NAMES[0].startswith("left_") else "right"


def hit_arm_joint_names(
  racket_hand: Literal["left", "right"] | None = None,
) -> tuple[str, ...]:
  """Wrist roll / pitch / yaw on the hitting side."""
  side = _hit_arm_side(racket_hand)
  return (
    f"{side}_wrist_roll_joint",
    f"{side}_wrist_pitch_joint",
    f"{side}_wrist_yaw_joint",
  )


def hit_arm_body_names(
  racket_hand: Literal["left", "right"] | None = None,
) -> tuple[str, ...]:
  """Shoulder / elbow / wrist bodies on the hitting side."""
  if racket_hand == "right":
    return _HIT_ARM_BODY_NAMES_RIGHT
  if racket_hand == "left":
    return _HIT_ARM_BODY_NAMES_LEFT
  return HIT_ARM_BODY_NAMES


def _hit_arm_wrist_dof_joint_names(
  racket_hand: Literal["left", "right"] | None = None,
) -> tuple[str, ...]:
  return hit_arm_joint_names(racket_hand)


def _hit_arm_dof_joint_names(
  racket_hand: Literal["left", "right"] | None = None,
) -> tuple[str, ...]:
  return hit_arm_joint_names(racket_hand)


def make_stage1_tracking_env_cfg() -> ManagerBasedRlEnvCfg:
  """Motion-only mimic cfg (no ball / landing commands)."""

  actor_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": "motion"}
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b, params={"command_name": "motion"}
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b, params={"command_name": "motion"}
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=0,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      motion_file="",
      anchor_body_name="",
      body_names=(),
    )
  }

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.08, 0.08),
          1: (-0.08, 0.08),
          2: (-0.08, 0.08),
        },
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        "kp_range": (0.95, 1.05),
        "kd_range": (0.95, 1.05),
        "operation": "scale",
      },
    ),
    "effort_limits": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        "effort_limit_range": (0.9, 1.0),
        "operation": "scale",
      },
    ),
  }

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=1,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "motion_wrist_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.0,
      params={
        "command_name": "motion",
        "std": 0.15,
        "joint_names": _hit_arm_wrist_dof_joint_names(),
      },
    ),
    "motion_keyframe_hit_arm_joint_pos": RewardTermCfg(
      func=mdp.motion_keyframe_relative_body_position_reward,
      weight=300.0,
      params={
        "command_name": "motion",
        "keyframe_times": HIT_ARM_KEYFRAME_TIMES_S,
        "body_names": HIT_ARM_BODY_NAMES,
        "std": 0.15,
        "window_s": 0.01,
      },
    ),
    "motion_keyframe_racket_site_pos": RewardTermCfg(
      func=mdp.motion_keyframe_racket_site_position_reward,
      weight=200.0,
      params={
        "command_name": "motion",
        "keyframe_times": HIT_ARM_KEYFRAME_TIMES_S,
        "site_name": "racket_point",
        "site_ref_body_name": next(n for n in HIT_ARM_BODY_NAMES if "wrist" in n),
        "std": 0.15,
        "window_s": 0.01,
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-2e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(
      func=mdp.motion_reached_last_frame,
      time_out=True,
      params={"command_name": "motion", "warmdown_s": 1.0},
    ),
    "anchor_pos": TerminationTermCfg(
      func=mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )


def apply_stage1_random_dt_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Enable env-sampled random motion ``dt`` and expose it as ``motion_dt_sample`` obs.

  Each control step samples ``delta_t`` uniformly in
  :attr:`MotionCommandCfg.dt_delta_range` inside :meth:`MotionCommand._update_command`.
  The policy action space stays **joint-only** (no ``motion_dt`` action term).
  """
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.dynamic_dt_enabled = True
  motion_cmd.random_dt_training_enabled = True
  dt_obs = ObservationTermCfg(
    func=mdp.motion_dt_sample,
    params={"command_name": "motion"},
  )
  for group_name in ("actor_stage1", "actor", "critic"):
    obs_group = cfg.observations.get(group_name)
    if obs_group is not None:
      obs_group.terms["motion_dt_sample"] = dt_obs
  cfg.scale_rewards_by_motion_dt = True
