import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    clip_actions: float | None = None,
  ):
    self.env = env
    self.clip_actions = clip_actions

    self.num_envs = self.unwrapped.num_envs
    self.device = torch.device(self.unwrapped.device)
    self.max_episode_length = self.unwrapped.max_episode_length
    self.num_actions = self.unwrapped.action_manager.total_action_dim
    self._modify_action_space()

    # Reset at the start since rsl_rl does not call reset.
    self.env.reset()

  def _collect_nonfinite_tensor(self, name: str, tensor: torch.Tensor, errors: list[str]) -> None:
    if tensor.numel() == 0:
      return
    mask = ~torch.isfinite(tensor)
    if not torch.any(mask):
      return
    if tensor.ndim == 0:
      errors.append(f"[NaN Breakpoint] non-finite in {name} (scalar)")
      return
    flat = mask.reshape(tensor.shape[0], -1)
    bad_ids = torch.nonzero(flat.any(dim=1), as_tuple=False).squeeze(-1)
    sample = bad_ids[:32].detach().cpu().tolist()
    errors.append(
      f"[NaN Breakpoint] non-finite in {name}, "
      f"env_ids={sample}, count={int(bad_ids.numel())}"
    )

  def _collect_nonfinite_nested(self, name: str, value, errors: list[str]) -> None:
    if isinstance(value, torch.Tensor):
      self._collect_nonfinite_tensor(name, value, errors)
      return
    if isinstance(value, TensorDict):
      for k, v in value.items():
        self._collect_nonfinite_nested(f"{name}.{k}", v, errors)
      return
    if isinstance(value, dict):
      for k, v in value.items():
        self._collect_nonfinite_nested(f"{name}.{k}", v, errors)
      return

  @property
  def cfg(self) -> ManagerBasedRlEnvCfg:
    return self.unwrapped.cfg

  @property
  def render_mode(self) -> str | None:
    return self.env.render_mode

  @property
  def observation_space(self) -> Space:
    return self.env.observation_space

  @property
  def action_space(self) -> Space:
    return self.env.action_space

  @classmethod
  def class_name(cls) -> str:
    return cls.__name__

  @property
  def unwrapped(self) -> ManagerBasedRlEnv:
    return self.env.unwrapped

  # Properties.

  @property
  def episode_length_buf(self) -> torch.Tensor:
    return self.unwrapped.episode_length_buf

  @episode_length_buf.setter
  def episode_length_buf(self, value: torch.Tensor) -> None:  # pyright: ignore[reportIncompatibleVariableOverride]
    self.unwrapped.episode_length_buf = value

  def seed(self, seed: int = -1) -> int:
    return self.unwrapped.seed(seed)

  def get_observations(self) -> TensorDict:
    obs_dict = self.unwrapped.observation_manager.compute()
    return TensorDict(obs_dict, batch_size=[self.num_envs])

  def reset(self) -> tuple[TensorDict, dict]:
    obs_dict, extras = self.env.reset()
    return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    errors: list[str] = []
    self._collect_nonfinite_tensor("actions", actions, errors)
    step_actions = actions
    # Keep stepping to inspect obs/reward even if action already has NaN/Inf.
    if errors:
      step_actions = torch.nan_to_num(step_actions, nan=0.0, posinf=0.0, neginf=0.0)
    if self.clip_actions is not None:
      step_actions = torch.clamp(step_actions, -self.clip_actions, self.clip_actions)
    obs_dict, rew, terminated, truncated, extras = self.env.step(step_actions)
    self._collect_nonfinite_nested("observations", obs_dict, errors)
    self._collect_nonfinite_tensor("reward", rew, errors)
    term_or_trunc = terminated | truncated
    assert isinstance(rew, torch.Tensor)
    assert isinstance(term_or_trunc, torch.Tensor)
    dones = term_or_trunc.to(dtype=torch.long)
    if not self.cfg.is_finite_horizon:
      extras["time_outs"] = truncated
    if errors:
      raise RuntimeError("\n".join(errors))
    return (
      TensorDict(obs_dict, batch_size=[self.num_envs]),
      rew,
      dones,
      extras,
    )

  def close(self) -> None:
    return self.env.close()

  # Private methods.

  def _modify_action_space(self) -> None:
    if self.clip_actions is None:
      return

    from mjlab.utils.spaces import Box, batch_space

    self.unwrapped.single_action_space = Box(
      shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
    )
    self.unwrapped.action_space = batch_space(
      self.unwrapped.single_action_space, self.num_envs
    )
