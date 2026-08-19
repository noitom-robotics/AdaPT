from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.adapt_tennis.mdp.commands import MotionCommandCfg


def apply_motion_reference_warmup(
  cfg: ManagerBasedRlEnvCfg,
  *,
  warmup_steps: int,
  step_source: str = "env",
  motion_command_name: str = "motion",
) -> None:
  """Keep motion reference at frame 0 until training passes ``warmup_steps``."""
  if warmup_steps <= 0:
    return
  if cfg.commands is None:
    raise ValueError("apply_motion_reference_warmup requires env_cfg.commands.")
  motion = cfg.commands.get(motion_command_name)
  if motion is None:
    raise ValueError(f"No command {motion_command_name!r} in env_cfg.commands.")
  if not isinstance(motion, MotionCommandCfg):
    raise TypeError(
      f"Command {motion_command_name!r} must be MotionCommandCfg, got {type(motion).__name__}."
    )
  if step_source not in ("env", "learning_iteration", "episode"):
    raise ValueError(
      "motion_warmup step_source must be 'env', 'learning_iteration', or 'episode', "
      f"got {step_source!r}."
    )
  motion.motion_warmup_steps = int(warmup_steps)
  motion.motion_warmup_step_source = step_source  # type: ignore[assignment]

  print(
    "[INFO] Motion reference warmup: hold frame 0 until "
    f"{step_source} step >= {warmup_steps}."
  )
