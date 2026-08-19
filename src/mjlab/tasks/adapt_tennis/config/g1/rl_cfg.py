"""RL configuration for Unitree G1 AdaPT Tennis stage-1 tracking."""

from dataclasses import replace

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_adapt_tennis_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO for stage-1 tracking with env-sampled random ``motion_dt`` (joint-only actions)."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      bound_loss_coef=0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_adapt_tennis_stage1_random_dt",
    save_interval=4000,
    num_steps_per_env=24,
    max_iterations=28_000,
  )


def unitree_g1_serve_tracking_stage1_random_dt_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Backward-compatible alias matching the original ServeTracking experiment name."""
  base = unitree_g1_adapt_tennis_ppo_runner_cfg()
  return replace(base, experiment_name="g1_serve_tracking_stage1_random_dt")
