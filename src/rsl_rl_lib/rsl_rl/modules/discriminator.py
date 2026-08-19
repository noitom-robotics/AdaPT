from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
    """AMP discriminator used to produce adversarial motion rewards."""

    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        device: str = "cpu",
        task_reward_lerp: float = 0.0,
    ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.task_reward_lerp = float(task_reward_lerp)

        layers: list[nn.Module] = []
        curr_in_dim = self.input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            curr_in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers).to(device)
        self.amp_linear = nn.Linear(curr_in_dim, 1).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(
        self,
        expert_amp_sequence: torch.Tensor,
        lambda_: float = 10.0,
    ) -> torch.Tensor:
        """Gradient penalty on discriminator input (concat of consecutive AMP frames)."""
        expert_data = expert_amp_sequence.clone().detach().requires_grad_(True)
        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return float(lambda_) * (grad.norm(2, dim=1) - 0.0).pow(2).mean()

    def predict_amp_reward(
        self,
        amp_sequence: torch.Tensor,
        task_reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return **unscaled** AMP shaping reward; multiply by ``amp_reward_coef`` only in AMPPPO.

        ``amp_sequence``: (B, amp_dim * num_frames).
        """
        with torch.no_grad():
            d = self.amp_linear(self.trunk(amp_sequence))
            # Same shaping as common AMP; ``amp_reward_coef`` is applied once in AMPPPO.process_env_step.
            reward = torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = (1.0 - self.task_reward_lerp) * reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
            reward = reward.squeeze(-1)
        return reward, d
