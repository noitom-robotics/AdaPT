"""Unitree G1 flat AdaPT Tennis stage-1 tracking environment configurations."""

from __future__ import annotations

from typing import Literal

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.adapt_tennis.mdp import MotionCommandCfg
from mjlab.tasks.adapt_tennis.stage1_tracking_env_cfg import (
  apply_stage1_random_dt_curriculum,
  hit_arm_body_names,
  hit_arm_joint_names,
  make_stage1_tracking_env_cfg,
)


def apply_racket_hand(
  cfg: ManagerBasedRlEnvCfg,
  racket_hand: Literal["left", "right"],
) -> None:
  """Switch G1 racket MJCF and hit-arm reward joints to ``left`` or ``right``."""
  cfg.scene.entities = {
    **(cfg.scene.entities or {}),
    "robot": get_g1_robot_cfg(racket_hand=racket_hand),
  }
  apply_racket_hand_rewards(cfg, racket_hand)


def apply_racket_hand_rewards(
  cfg: ManagerBasedRlEnvCfg,
  racket_hand: Literal["left", "right"],
) -> None:
  joints = hit_arm_joint_names(racket_hand)
  bodies = hit_arm_body_names(racket_hand)
  wrist_body = next(n for n in bodies if "wrist" in n)

  wrist_term = cfg.rewards.get("motion_wrist_joint_pos") if cfg.rewards else None
  if wrist_term is not None and "joint_names" in wrist_term.params:
    wrist_term.params["joint_names"] = joints

  hit_arm_term = (
    cfg.rewards.get("motion_keyframe_hit_arm_joint_pos") if cfg.rewards else None
  )
  if hit_arm_term is not None and "body_names" in hit_arm_term.params:
    hit_arm_term.params["body_names"] = bodies

  racket_site_term = (
    cfg.rewards.get("motion_keyframe_racket_site_pos") if cfg.rewards else None
  )
  if racket_site_term is not None and "site_ref_body_name" in racket_site_term.params:
    racket_site_term.params["site_ref_body_name"] = wrist_body


def _self_collision_sensor_cfg() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def _racket_ground_contact_sensor_cfg() -> ContactSensorCfg:
  """Contact between racket collision proxy and plane terrain."""
  return ContactSensorCfg(
    name="racket_ground_collision",
    primary=ContactMatch(
      mode="geom",
      pattern="tennis_racket_collision",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  racket_hand: Literal["left", "right"] = "left",
) -> ManagerBasedRlEnvCfg:
  """G1 motion-mimic only: random motion-``dt`` curriculum, no ball / gripper.

  Args:
    racket_hand: ``left`` uses ``G1_LEFT_XML``, ``right`` uses ``G1_RIGHT_XML``.
      Defaults to left to match ``HIT_ARM_BODY_NAMES`` in stage-1 tracking.
  """
  cfg = make_stage1_tracking_env_cfg()
  apply_stage1_random_dt_curriculum(cfg)

  cfg.scene.entities = {
    "robot": get_g1_robot_cfg(racket_hand=racket_hand)
  }
  apply_racket_hand_rewards(cfg, racket_hand)
  cfg.scene.sensors = (
    _self_collision_sensor_cfg(),
    _racket_ground_contact_sensor_cfg(),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params[
      "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  if "ee_body_pos" in cfg.terminations:
    cfg.terminations["ee_body_pos"].params["body_names"] = (
      "left_ankle_roll_link",
      "right_ankle_roll_link",
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
    )

  cfg.viewer.body_name = "torso_link"

  if not has_state_estimation:
    actor_group_name = (
      "actor"
      if "actor" in cfg.observations
      else "actor_stage1"
      if "actor_stage1" in cfg.observations
      else None
    )
    if actor_group_name is not None:
      new_actor_terms = {
        k: v
        for k, v in cfg.observations[actor_group_name].terms.items()
        if k not in ["motion_anchor_pos_b", "base_lin_vel"]
      }
      cfg.observations[actor_group_name] = ObservationGroupCfg(
        terms=new_actor_terms,
        concatenate_terms=True,
        enable_corruption=True,
        history_length=cfg.observations[actor_group_name].history_length,
      )

  if play:
    cfg.episode_length_s = int(1e9)

    if "actor" in cfg.observations:
      cfg.observations["actor"].enable_corruption = False
    elif "actor_stage1" in cfg.observations:
      cfg.observations["actor_stage1"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
