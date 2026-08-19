"""OmniXtreme / Table IX asymmetric torque-speed actuator model.

Implements the operating envelope and friction model from OmniXtreme
(arxiv:2602.23843), equations (4)-(6):

  tau_max,0 = tau_y1 if v * tau_in > 0 else tau_y2
  |tau| <= f(|v|) * tau_max,0   (piecewise in vx1, vx2)
  tau_applied = tau_clipped - (mu_s * tanh(v / v_act) + mu_d * v)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity

OmniExtremeCfgT = TypeVar("OmniExtremeCfgT", bound="OmniExtremeActuatorCfg")


@dataclass(frozen=True, slots=True)
class OmniExtremeActuatorParams:
  """Physical parameters from OmniXtreme Table IX (per motor, joint space)."""

  tau_y1: float
  """Peak motoring torque (Nm): torque and velocity same sign."""
  tau_y2: float
  """Peak braking/regenerative torque (Nm): torque and velocity opposite sign."""
  vx1: float
  """Speed (rad/s) where torque envelope begins to decrease."""
  vx2: float
  """Speed (rad/s) where achievable torque reaches zero."""
  mu_s: float
  """Coulomb friction magnitude (Nm)."""
  v_act: float
  """Tanh activation velocity for Coulomb friction (rad/s)."""
  mu_d: float
  """Viscous friction coefficient (Nm·s/rad)."""
  armature: float
  """Reflected rotor inertia (kg·m²)."""


# OmniXtreme Table IX presets for Unitree G1 motors.
TABLE_IX_5020_16 = OmniExtremeActuatorParams(
  tau_y1=24.8,
  tau_y2=31.9,
  vx1=30.86,
  vx2=40.13,
  mu_s=0.6,
  v_act=0.01,
  mu_d=0.06,
  armature=3.610e-3,
)
TABLE_IX_7520_14 = OmniExtremeActuatorParams(
  tau_y1=71.0,
  tau_y2=83.3,
  vx1=22.63,
  vx2=35.52,
  mu_s=1.6,
  v_act=0.01,
  mu_d=0.16,
  armature=1.018e-2,
)
TABLE_IX_7520_22 = OmniExtremeActuatorParams(
  tau_y1=111.0,
  tau_y2=131.0,
  vx1=14.5,
  vx2=22.7,
  mu_s=2.4,
  v_act=0.01,
  mu_d=0.24,
  armature=2.510e-2,
)
TABLE_IX_4010_25 = OmniExtremeActuatorParams(
  tau_y1=4.8,
  tau_y2=8.6,
  vx1=15.3,
  vx2=24.76,
  mu_s=0.6,
  v_act=0.01,
  mu_d=0.06,
  armature=4.250e-3,
)


def omni_extreme_torque_envelope(
  vel: torch.Tensor,
  tau_y1: torch.Tensor,
  tau_y2: torch.Tensor,
  vx1: torch.Tensor,
  vx2: torch.Tensor,
  *,
  velocity_epsilon: float = 1e-3,
  torque_scale: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return ``(tau_low, tau_high)`` admissible torque bounds for joint velocity ``vel``.

  Matches the reference OmniXtreme ``deploy_mujoco.py`` envelope logic.
  """
  eps = 1e-6
  abs_vel = vel.abs()
  over = torch.clamp(abs_vel - vx1, min=0.0)
  denom = torch.clamp(vx2 - vx1, min=eps)

  y1 = tau_y1 * torque_scale
  y2 = tau_y2 * torque_scale

  base_pos = torch.where(
    abs_vel <= velocity_epsilon,
    y2,
    torch.where(vel >= 0.0, y1, y2),
  )
  slope_pos = base_pos / denom
  tau_high = torch.clamp(base_pos - slope_pos * over, min=0.0)

  base_neg = torch.where(
    abs_vel <= velocity_epsilon,
    -y2,
    torch.where(vel >= 0.0, -y2, -y1),
  )
  slope_neg = (-base_neg) / denom
  tau_low = torch.clamp(base_neg + slope_neg * over, max=0.0)
  return tau_low, tau_high


def omni_extreme_friction(
  vel: torch.Tensor,
  mu_s: torch.Tensor,
  v_act: torch.Tensor,
  mu_d: torch.Tensor,
) -> torch.Tensor:
  """Friction torque subtracted after envelope clipping (Eq. 6)."""
  return mu_s * torch.tanh(vel / torch.clamp(v_act, min=1e-8)) + mu_d * vel


@dataclass(kw_only=True)
class OmniExtremeActuatorCfg(IdealPdActuatorCfg):
  """Ideal PD actuator with OmniXtreme Table IX torque-speed envelope + friction."""

  params: OmniExtremeActuatorParams
  """Motor parameters (Table IX row)."""

  torque_scale: float = 1.0
  """Scale on ``tau_y1``/``tau_y2`` (e.g. 2.0 for dual-5020 ankle linkage)."""

  velocity_epsilon: float = 1e-3
  """Treat ``|v| <= velocity_epsilon`` as zero velocity for Y1/Y2 selection."""

  apply_friction: bool = True
  """Subtract Coulomb + viscous friction after clipping."""

  # TEMP: disable OmniXtreme envelope + friction; clip with constant effort_limit.
  use_torque_envelope: bool = False
  """When False, skip envelope/friction and clamp with ``effort_limit`` like IdealPd."""

  def __post_init__(self) -> None:
    if self.use_torque_envelope and self.effort_limit != float("inf"):
      import warnings

      warnings.warn(
        f"{self.__class__.__name__}: effort_limit={self.effort_limit} is ignored; "
        "torque limits come from the OmniXtreme envelope (tau_y1/tau_y2, vx1/vx2).",
        UserWarning,
        stacklevel=2,
      )
    super().__post_init__()

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> OmniExtremeActuator:
    return OmniExtremeActuator(self, entity, target_ids, target_names)


class OmniExtremeActuator(IdealPdActuator[OmniExtremeCfgT], Generic[OmniExtremeCfgT]):
  """PD actuator with velocity-dependent asymmetric torque limits and friction."""

  def __init__(
    self,
    cfg: OmniExtremeCfgT,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._tau_y1: torch.Tensor | None = None
    self._tau_y2: torch.Tensor | None = None
    self._vx1: torch.Tensor | None = None
    self._vx2: torch.Tensor | None = None
    self._mu_s: torch.Tensor | None = None
    self._v_act: torch.Tensor | None = None
    self._mu_d: torch.Tensor | None = None
    self._joint_vel: torch.Tensor | None = None

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    num_envs = data.nworld
    num_joints = len(self._target_names)
    p = self.cfg.params

    def _full(value: float) -> torch.Tensor:
      return torch.full((num_envs, num_joints), value, dtype=torch.float, device=device)

    self._tau_y1 = _full(p.tau_y1)
    self._tau_y2 = _full(p.tau_y2)
    self._vx1 = _full(p.vx1)
    self._vx2 = _full(p.vx2)
    self._mu_s = _full(p.mu_s)
    self._v_act = _full(p.v_act)
    self._mu_d = _full(p.mu_d)
    self._joint_vel = torch.zeros(num_envs, num_joints, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    if not self.cfg.use_torque_envelope:
      return IdealPdActuator.compute(self, cmd)

    assert self.stiffness is not None
    assert self.damping is not None
    self._joint_vel = cmd.vel

    pos_error = cmd.position_target - cmd.pos
    vel_error = cmd.velocity_target - cmd.vel
    tau_in = self.stiffness * pos_error + self.damping * vel_error + cmd.effort_target
    return self._clip_effort(tau_in)

  def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
    if not self.cfg.use_torque_envelope:
      return IdealPdActuator._clip_effort(self, effort)

    assert self._joint_vel is not None
    assert self._tau_y1 is not None
    assert self._tau_y2 is not None
    assert self._vx1 is not None
    assert self._vx2 is not None
    assert self._mu_s is not None
    assert self._v_act is not None
    assert self._mu_d is not None

    tau_low, tau_high = omni_extreme_torque_envelope(
      self._joint_vel,
      self._tau_y1,
      self._tau_y2,
      self._vx1,
      self._vx2,
      velocity_epsilon=self.cfg.velocity_epsilon,
      torque_scale=self.cfg.torque_scale,
    )
    tau_clipped = torch.clamp(effort, min=tau_low, max=tau_high)

    if not self.cfg.apply_friction:
      return tau_clipped

    friction = omni_extreme_friction(
      self._joint_vel, self._mu_s, self._v_act, self._mu_d
    )
    return tau_clipped - friction
