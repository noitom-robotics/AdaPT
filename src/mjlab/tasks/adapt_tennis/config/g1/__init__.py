from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.adapt_tennis.rl import AdaPTTennisOnPolicyRunner

from .env_cfgs import unitree_g1_flat_tracking_env_cfg
from .rl_cfg import (
  unitree_g1_adapt_tennis_ppo_runner_cfg,
  unitree_g1_serve_tracking_stage1_random_dt_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-AdaPT-Tennis-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_adapt_tennis_ppo_runner_cfg(),
  runner_cls=AdaPTTennisOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-AdaPT-Tennis-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_adapt_tennis_ppo_runner_cfg(),
  runner_cls=AdaPTTennisOnPolicyRunner,
)

# Backward-compatible alias of the original ServeTracking stage-1 random-dt task.
register_mjlab_task(
  task_id="Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_serve_tracking_stage1_random_dt_ppo_runner_cfg(),
  runner_cls=AdaPTTennisOnPolicyRunner,
)
