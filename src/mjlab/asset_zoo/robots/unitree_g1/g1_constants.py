"""Unitree G1 constants."""

from pathlib import Path
from typing import Literal

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.spec_config import CollisionCfg

RacketHand = Literal["left", "right"]

##
# MJCF and assets.
##

G1_LEFT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1_leftracket.xml"
)
G1_RIGHT_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1_rightracket.xml"
)
assert G1_LEFT_XML.exists()
assert G1_RIGHT_XML.exists()


def get_spec(racket_hand: RacketHand = "right") -> mujoco.MjSpec:
  xml = G1_LEFT_XML if racket_hand == "left" else G1_RIGHT_XML
  return mujoco.MjSpec.from_file(str(xml))


##
# Actuator config.
#
# Effort / armature follow the G1 MJCF defaults (g1_{left,right}racket.xml):
#   hip_pitch / hip_roll / knee: 139 Nm, armature 0.025101925  (7520-22)
#   hip_yaw / waist_yaw:         88 Nm,  armature 0.01017752004 (7520-14)
#   ankle pitch/roll:            50 Nm,  armature 0.00721945    (2x 5020)
#   arm 5020 joints:             25 Nm,  armature 0.003609725
#   wrist pitch/yaw:              5 Nm,  armature 0.00425       (4010)
# Kp/Kd are derived from the same reflected inertia so they stay consistent
# with the XML armature (``k = I * ω_n^2``, ``d = 2 ζ I ω_n``).
##

ROTOR_INERTIAS_5020 = (
  0.139e-4,
  0.017e-4,
  0.169e-4,
)
GEARS_5020 = (
  1,
  1 + (46 / 18),
  1 + (56 / 16),
)
ARMATURE_5020 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_5020, GEARS_5020
)

ROTOR_INERTIAS_7520_14 = (
  0.489e-4,
  0.098e-4,
  0.533e-4,
)
GEARS_7520_14 = (
  1,
  4.5,
  1 + (48 / 22),
)
ARMATURE_7520_14 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_14, GEARS_7520_14
)

ROTOR_INERTIAS_7520_22 = (
  0.489e-4,
  0.109e-4,
  0.738e-4,
)
GEARS_7520_22 = (
  1,
  4.5,
  5,
)
ARMATURE_7520_22 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_7520_22, GEARS_7520_22
)

ROTOR_INERTIAS_4010 = (
  0.068e-4,
  0.0,
  0.0,
)
GEARS_4010 = (
  1,
  5,
  5,
)
ARMATURE_4010 = reflected_inertia_from_two_stage_planetary(
  ROTOR_INERTIAS_4010, GEARS_4010
)

ACTUATOR_5020 = ElectricActuator(
  reflected_inertia=ARMATURE_5020,
  velocity_limit=37.0,
  effort_limit=25.0,
)
ACTUATOR_7520_14 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_14,
  velocity_limit=32.0,
  effort_limit=88.0,
)
ACTUATOR_7520_22 = ElectricActuator(
  reflected_inertia=ARMATURE_7520_22,
  velocity_limit=20.0,
  effort_limit=139.0,
)
ACTUATOR_4010 = ElectricActuator(
  reflected_inertia=ARMATURE_4010,
  velocity_limit=22.0,
  effort_limit=5.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

G1_ACTUATOR_5020 = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_elbow_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_wrist_roll_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=ACTUATOR_5020.effort_limit,
  armature=ACTUATOR_5020.reflected_inertia,
)
G1_ACTUATOR_7520_14 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=ACTUATOR_7520_14.effort_limit,
  armature=ACTUATOR_7520_14.reflected_inertia,
)
G1_ACTUATOR_7520_22 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=ACTUATOR_7520_22.effort_limit,
  armature=ACTUATOR_7520_22.reflected_inertia,
)
G1_ACTUATOR_4010 = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=ACTUATOR_4010.effort_limit,
  armature=ACTUATOR_4010.reflected_inertia,
)

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuators.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuators' armatures.
# The 27-DoF racket XMLs lock waist pitch/roll; keep the cfg for completeness.
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=STIFFNESS_5020 * 2,
  damping=DAMPING_5020 * 2,
  effort_limit=ACTUATOR_5020.effort_limit * 2,
  armature=ACTUATOR_5020.reflected_inertia * 2,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.783675),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_joint": 1.28,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.76),
  joint_pos={
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# MuJoCo contype/conaffinity bit masks (pair collides iff either bitwise AND is nonzero).
# - 1: default robot / ground contacts
# - 2: racket proxy layer (excludes ball below)
# - 4: tennis ball (does not pair with racket layer)
_COLLISION_MASK_DEFAULT = 1
_COLLISION_MASK_RACKET = 2
RACKET_COLLISION_CONTYPE = _COLLISION_MASK_RACKET
RACKET_COLLISION_CONAFFINITY = _COLLISION_MASK_DEFAULT | _COLLISION_MASK_RACKET

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
# ``tennis_racket_collision`` uses a separate layer so it still hits ground/body but not
# ``tennis_ball_geom`` (ball must use contype=4 conaffinity=1 in tennis_ball.xml).
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype={
    r"^tennis_racket_collision$": RACKET_COLLISION_CONTYPE,
    ".*_collision": _COLLISION_MASK_DEFAULT,
  },
  conaffinity={
    r"^tennis_racket_collision$": RACKET_COLLISION_CONAFFINITY,
    ".*_collision": _COLLISION_MASK_DEFAULT,
  },
  condim={
    r"^(left|right)_foot[1-7]_collision$": 3,
    ".*_collision": 1,
  },
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={
    r"^(left|right)_foot[1-7]_collision$": (0.6,),
  },
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def _action_scale_for_articulation(
  articulation: EntityArticulationInfoCfg,
) -> dict[str, float]:
  """Compute ``0.25 * effort_limit / stiffness`` for builtin position actuators."""
  scale: dict[str, float] = {}
  for actuator in articulation.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    s = actuator.stiffness
    e = actuator.effort_limit
    assert e is not None
    for name in actuator.target_names_expr:
      scale[name] = 0.25 * e / s
  return scale


def _effort_limit_dict_for_articulation(
  articulation: EntityArticulationInfoCfg,
) -> dict[str, float]:
  """Map joint name patterns to constant effort limits (Nm) from builtin actuator cfg."""
  limits: dict[str, float] = {}
  for actuator in articulation.actuators:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    e = actuator.effort_limit
    assert e is not None
    for name in actuator.target_names_expr:
      limits[name] = float(e)
  return limits


def g1_builtin_effort_limits_for_joint_names(joint_names: list[str]) -> list[float]:
  """Per-joint builtin effort limits (Nm) in ``joint_names`` order (for deploy / export)."""
  from mjlab.utils.lab_api.string import resolve_matching_names_values

  _, _, values = resolve_matching_names_values(
    _effort_limit_dict_for_articulation(G1_ARTICULATION), joint_names
  )
  return [float(v) for v in values]


def get_g1_robot_cfg(*, racket_hand: RacketHand = "right") -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  ``racket_hand`` selects ``G1_LEFT_XML`` or ``G1_RIGHT_XML``.
  """
  hand = racket_hand
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=lambda: get_spec(hand),
    articulation=G1_ARTICULATION,
  )


G1_ACTION_SCALE: dict[str, float] = _action_scale_for_articulation(G1_ARTICULATION)


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_g1_robot_cfg())

  viewer.launch(robot.spec.compile())
