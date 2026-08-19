from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import Discriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class AMPPPO(PPO):
    """PPO + AMP reward/discriminator (adapted to current rsl_rl framework)."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: Discriminator,
        amp_reward_coef: float = 1.0,
        amp_loss_coef: float = 1.0,
        amp_grad_pen_coef: float = 10.0,
        amp_obs_group: str = "amp",
        amp_buffer_size: int = 100000,
        amp_discriminator_batch_size: int = 4096,
        amp_discriminator_updates: int = 1,
        amp_task_reward_lerp: float = 0.0,
        amp_num_frames: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(actor=actor, critic=critic, storage=storage, **kwargs)
        self.discriminator = discriminator.to(self.device)
        self.amp_reward_coef = float(amp_reward_coef)
        self.amp_loss_coef = float(amp_loss_coef)
        self.amp_grad_pen_coef = float(amp_grad_pen_coef)
        self.amp_obs_group = amp_obs_group
        self.amp_discriminator_batch_size = int(amp_discriminator_batch_size)
        self.amp_discriminator_updates = int(amp_discriminator_updates)
        self.amp_task_reward_lerp = float(amp_task_reward_lerp)
        self.amp_num_frames = max(1, int(amp_num_frames))
        self._amp_obs_history: torch.Tensor | None = None
        self._amp_history_filled = 0
        self.amp_rewards: torch.Tensor | None = None
        self._amp_reward_running_sum = 0.0
        self._amp_reward_running_count = 0
        self._amp_buffer_size = int(amp_buffer_size)
        self._amp_rb_sequences: torch.Tensor | None = None
        self._amp_rb_step = 0
        self._amp_rb_num_samples = 0
        self._amp_expert_sampler: Callable[[int], torch.Tensor | None] | None = None
        self._disc_optimizer = optim.Adam(
            [
                {"params": self.discriminator.trunk.parameters(), "weight_decay": 1.0e-3},
                {"params": self.discriminator.amp_linear.parameters(), "weight_decay": 1.0e-2},
            ],
            lr=self.learning_rate,
        )

    def _get_amp_obs(self, obs: TensorDict) -> torch.Tensor | None:
        if self.amp_obs_group in obs.keys():
            amp_obs = obs[self.amp_obs_group]
            if amp_obs.ndim == 1:
                amp_obs = amp_obs.unsqueeze(0)
            return amp_obs
        return None

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        amp_obs = self._get_amp_obs(obs)
        if amp_obs is not None:
            if (
                self._amp_obs_history is None
                or self._amp_obs_history.shape[1] != amp_obs.shape[0]
                or self._amp_obs_history.shape[2] != amp_obs.shape[1]
            ):
                self._amp_obs_history = amp_obs.detach().clone().unsqueeze(0).repeat(self.amp_num_frames, 1, 1)
                self._amp_history_filled = 1
            else:
                self._amp_obs_history[:-1] = self._amp_obs_history[1:].clone()
                self._amp_obs_history[-1] = amp_obs.detach()
                self._amp_history_filled = min(self._amp_history_filled + 1, self.amp_num_frames)

            if self._amp_history_filled >= self.amp_num_frames:
                amp_sequence = self._amp_obs_history.permute(1, 0, 2).reshape(amp_obs.shape[0], -1)
                amp_reward, _ = self.discriminator.predict_amp_reward(amp_sequence, rewards)
                self.amp_rewards = amp_reward.detach()
                self._amp_reward_running_sum += float(torch.mean(amp_reward).item())
                self._amp_reward_running_count += 1
                rewards = (1-self.amp_reward_coef) * rewards + self.amp_reward_coef * amp_reward
                self._insert_amp_sequences(amp_sequence.detach())

            if self._amp_obs_history is not None and dones.numel() > 0:
                done_mask = dones.view(-1).to(dtype=torch.bool)
                if torch.any(done_mask):
                    done_obs = amp_obs.detach()[done_mask].unsqueeze(0).repeat(self.amp_num_frames, 1, 1)
                    self._amp_obs_history[:, done_mask] = done_obs
        super().process_env_step(obs, rewards, dones, extras)

    def _insert_amp_sequences(self, sequences: torch.Tensor) -> None:
        num_rows = int(sequences.shape[0])
        if num_rows == 0:
            return
        if self._amp_rb_sequences is None:
            seq_dim = int(sequences.shape[-1])
            self._amp_rb_sequences = torch.zeros(
                self._amp_buffer_size, seq_dim, device=sequences.device, dtype=sequences.dtype
            )

        start_idx = self._amp_rb_step
        end_idx = start_idx + num_rows
        if end_idx > self._amp_buffer_size:
            first_len = self._amp_buffer_size - start_idx
            self._amp_rb_sequences[start_idx:self._amp_buffer_size] = sequences[:first_len]
            remain = end_idx - self._amp_buffer_size
            self._amp_rb_sequences[:remain] = sequences[first_len:]
        else:
            self._amp_rb_sequences[start_idx:end_idx] = sequences

        self._amp_rb_num_samples = min(self._amp_buffer_size, max(end_idx, self._amp_rb_num_samples))
        self._amp_rb_step = (self._amp_rb_step + num_rows) % self._amp_buffer_size

    def _sample_amp_sequences(self, batch_size: int) -> torch.Tensor | None:
        if self._amp_rb_sequences is None or int(self._amp_rb_num_samples) <= 0:
            return None
        n = int(self._amp_rb_num_samples)
        if n == 0:
            return None
        bs = min(int(batch_size), n)
        idx = torch.randint(0, n, (bs,), device=self._amp_rb_sequences.device)
        return self._amp_rb_sequences[idx]

    def set_amp_expert_sampler(self, sampler: Callable[[int], torch.Tensor | None] | None) -> None:
        """Register expert sampler: batch_size -> (B, amp_dim * num_frames) or None."""
        self._amp_expert_sampler = sampler

    def _update_discriminator(self) -> dict[str, float]:
        if self.amp_discriminator_updates <= 0:
            return {"amp_loss": 0.0, "amp_grad_pen": 0.0, "amp_policy_pred": 0.0, "amp_expert_pred": 0.0}
        mean_amp_loss = 0.0
        mean_grad_pen = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        num = 0
        for _ in range(self.amp_discriminator_updates):
            sample_policy = self._sample_amp_sequences(self.amp_discriminator_batch_size)
            if sample_policy is None or self._amp_expert_sampler is None:
                continue
            sample_expert = self._amp_expert_sampler(self.amp_discriminator_batch_size)
            if sample_expert is None:
                continue
            policy_d = self.discriminator(sample_policy)
            expert_d = self.discriminator(sample_expert)
            expert_loss = torch.nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = torch.nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen = self.discriminator.compute_grad_pen(sample_expert, lambda_=self.amp_grad_pen_coef)
            loss = self.amp_loss_coef * (amp_loss + grad_pen)
            self._disc_optimizer.zero_grad()
            loss.backward()
            self._disc_optimizer.step()
            mean_amp_loss += float(amp_loss.item())
            mean_grad_pen += float(grad_pen.item())
            mean_policy_pred += float(policy_d.mean().item())
            mean_expert_pred += float(expert_d.mean().item())
            num += 1
        if num == 0:
            return {"amp_loss": 0.0, "amp_grad_pen": 0.0, "amp_policy_pred": 0.0, "amp_expert_pred": 0.0}
        return {
            "amp_loss": mean_amp_loss / num,
            "amp_grad_pen": mean_grad_pen / num,
            "amp_policy_pred": mean_policy_pred / num,
            "amp_expert_pred": mean_expert_pred / num,
        }

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        disc_dict = self._update_discriminator()
        loss_dict.update(disc_dict)
        if self._amp_reward_running_count > 0:
            loss_dict["amp_reward"] = self._amp_reward_running_sum / float(self._amp_reward_running_count)
        else:
            loss_dict["amp_reward"] = 0.0
        self._amp_reward_running_sum = 0.0
        self._amp_reward_running_count = 0
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def save(self) -> dict:
        saved = super().save()
        saved["discriminator_state_dict"] = self.discriminator.state_dict()
        saved["discriminator_optimizer_state_dict"] = self._disc_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if "discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        if "discriminator_optimizer_state_dict" in loaded_dict and load_cfg and load_cfg.get("optimizer", True):
            self._disc_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMPPPO":
        alg_class: type[AMPPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])
        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore
        actor_custom_cfg = actor_cfg.pop("custom_cfg", {}) or {}
        critic_custom_cfg = critic_cfg.pop("custom_cfg", {}) or {}
        # Remove cfg fields that are valid at config level but unsupported by MLPModel ctor.
        for k in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            actor_cfg.pop(k, None)
            critic_cfg.pop(k, None)

        # "amp" is an observation set name; amp_obs_group is the concrete TensorDict key (e.g. "amp_obs").
        default_sets = ["actor", "critic", "amp"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg, **actor_custom_cfg).to(device)
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            critic_cfg["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg, **critic_custom_cfg).to(device)
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        amp_group = cfg["algorithm"].get("amp_obs_group", "amp_obs")
        if amp_group not in obs.keys():
            raise KeyError(f"AMP obs group '{amp_group}' not found in observations.")
        amp_dim = int(obs[amp_group].shape[-1])
        amp_num_frames = max(1, int(cfg["algorithm"].get("amp_num_frames", 1)))
        disc_hidden = cfg["algorithm"].pop("amp_discriminator_hidden_dims", [512, 256])
        disc = Discriminator(
            input_dim=amp_dim * amp_num_frames,
            hidden_layer_sizes=list(disc_hidden),
            device=device,
            task_reward_lerp=float(cfg["algorithm"].get("amp_task_reward_lerp", 0.0)),
        ).to(device)

        return alg_class(
            actor=actor,
            critic=critic,
            storage=storage,
            discriminator=disc,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
