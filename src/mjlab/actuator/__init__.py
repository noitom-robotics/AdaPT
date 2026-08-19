"""Actuator implementations for mjlab."""

from mjlab.actuator.actuator import Actuator as Actuator
from mjlab.actuator.actuator import ActuatorCfg as ActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd as ActuatorCmd
from mjlab.actuator.actuator import CommandField as CommandField
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator as BuiltinMotorActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuatorCfg as BuiltinMotorActuatorCfg,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinMuscleActuator as BuiltinMuscleActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinMuscleActuatorCfg as BuiltinMuscleActuatorCfg,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinPositionActuator as BuiltinPositionActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinPositionActuatorCfg as BuiltinPositionActuatorCfg,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinVelocityActuator as BuiltinVelocityActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinVelocityActuatorCfg as BuiltinVelocityActuatorCfg,
)
from mjlab.actuator.builtin_group import BuiltinActuatorGroup as BuiltinActuatorGroup
from mjlab.actuator.dc_actuator import DcMotorActuator as DcMotorActuator
from mjlab.actuator.dc_actuator import DcMotorActuatorCfg as DcMotorActuatorCfg
from mjlab.actuator.learned_actuator import LearnedMlpActuator as LearnedMlpActuator
from mjlab.actuator.learned_actuator import (
  LearnedMlpActuatorCfg as LearnedMlpActuatorCfg,
)
from mjlab.actuator.omni_extreme_actuator import (
  OmniExtremeActuator as OmniExtremeActuator,
)
from mjlab.actuator.omni_extreme_actuator import (
  OmniExtremeActuatorCfg as OmniExtremeActuatorCfg,
)
from mjlab.actuator.omni_extreme_actuator import (
  OmniExtremeActuatorParams as OmniExtremeActuatorParams,
)
from mjlab.actuator.omni_extreme_actuator import (
  TABLE_IX_4010_25 as TABLE_IX_4010_25,
)
from mjlab.actuator.omni_extreme_actuator import (
  TABLE_IX_5020_16 as TABLE_IX_5020_16,
)
from mjlab.actuator.omni_extreme_actuator import (
  TABLE_IX_7520_14 as TABLE_IX_7520_14,
)
from mjlab.actuator.omni_extreme_actuator import (
  TABLE_IX_7520_22 as TABLE_IX_7520_22,
)
from mjlab.actuator.omni_extreme_actuator import (
  omni_extreme_friction as omni_extreme_friction,
)
from mjlab.actuator.omni_extreme_actuator import (
  omni_extreme_torque_envelope as omni_extreme_torque_envelope,
)
from mjlab.actuator.pd_actuator import IdealPdActuator as IdealPdActuator
from mjlab.actuator.pd_actuator import IdealPdActuatorCfg as IdealPdActuatorCfg
from mjlab.actuator.xml_actuator import XmlActuator as XmlActuator
from mjlab.actuator.xml_actuator import XmlActuatorCfg as XmlActuatorCfg
