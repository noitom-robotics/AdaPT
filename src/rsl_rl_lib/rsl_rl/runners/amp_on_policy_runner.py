from __future__ import annotations

from math import prod

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class AmpOnPolicyRunner(OnPolicyRunner):
    """AMP runner: expert samples are ``num_frames`` consecutive AMP observations (concat)."""

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        # Accept extra runner kwargs (e.g. registry_name) for compatibility with mjlab train launcher.
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        self._amp_term_names, self._amp_term_dims = self._resolve_amp_term_spec()
        self._amp_num_frames = max(1, int(getattr(self.alg, "amp_num_frames", 1)))
        if hasattr(self.alg, "set_amp_expert_sampler"):
            self.alg.set_amp_expert_sampler(self._sample_expert_amp_sequences)

    def _sample_expert_amp_sequences(self, batch_size: int) -> torch.Tensor | None:
        """Sample expert AMP sequences: concat of ``amp_num_frames`` consecutive motion frames."""
        env = self.env.unwrapped
        motion_cmd = env.command_manager.get_term("motion")
        motion = motion_cmd.motion
        lengths = motion.lengths
        if lengths.numel() == 0:
            return None

        device = lengths.device
        bs = int(batch_size)
        motion_ids = torch.randint(0, motion.num_motions, (bs,), device=device)
        clip_lens = lengths[motion_ids]
        # Consecutive frames t, t+1, ..., t + num_frames - 1 (same layout as policy-side buffer concat).
        max_t = torch.clamp(clip_lens - self._amp_num_frames, min=0)
        t = torch.floor(torch.rand(bs, device=device) * (max_t.to(torch.float32) + 1.0)).to(torch.long)
        offsets = torch.arange(self._amp_num_frames, device=device).unsqueeze(0)
        all_steps = t.unsqueeze(1) + offsets
        all_steps = torch.minimum(all_steps, (clip_lens - 1).unsqueeze(1))

        frames: list[torch.Tensor] = []
        for i in range(self._amp_num_frames):
            frames.append(self._build_amp_obs_from_motion(motion_cmd, motion_ids, all_steps[:, i]))
        seq = torch.cat(frames, dim=-1)
        expected_dim = int(getattr(self.alg.discriminator, "input_dim", seq.shape[-1]))
        if seq.shape[-1] != expected_dim:
            raise ValueError(
                "AMP expert sequence dim mismatch: "
                f"got={seq.shape[-1]}, expected={expected_dim}, "
                f"amp_terms={self._amp_term_names}, amp_num_frames={self._amp_num_frames}"
            )
        return seq.to(self.device)

    def _build_amp_obs_from_motion(self, motion_cmd, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        body_pos_w = motion_cmd.motion.gather_body_pos_w(motion_ids, time_steps)
        body_quat_w = motion_cmd.motion.gather_body_quat_w(motion_ids, time_steps)
        body_lin_vel_w = motion_cmd.motion.gather_body_lin_vel_w(motion_ids, time_steps)
        body_ang_vel_w = motion_cmd.motion.gather_body_ang_vel_w(motion_ids, time_steps)
        joint_pos_motion = motion_cmd.motion.gather_joint_pos(motion_ids, time_steps)
        joint_vel_motion = motion_cmd.motion.gather_joint_vel(motion_ids, time_steps)
        num_bodies = body_pos_w.shape[1]
        anchor_idx = int(motion_cmd.motion_anchor_body_index)
        anchor_pos_w = body_pos_w[:, anchor_idx].unsqueeze(1).repeat(1, num_bodies, 1)
        anchor_quat_w = body_quat_w[:, anchor_idx].unsqueeze(1).repeat(1, num_bodies, 1)
        body_pos_b, body_quat_b = subtract_frame_transforms(anchor_pos_w, anchor_quat_w, body_pos_w, body_quat_w)
        body_ori_b = matrix_from_quat(body_quat_b)[..., :2].reshape(body_pos_w.shape[0], -1)
        body_lin_vel_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_lin_vel_w.reshape(-1, 3)).reshape(
            body_pos_w.shape[0], num_bodies, 3
        )
        body_ang_vel_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), body_ang_vel_w.reshape(-1, 3)).reshape(
            body_pos_w.shape[0], num_bodies, 3
        )
        joint_pos = self._build_joint_rel_obs_from_motion(motion_cmd, joint_pos_motion)
        joint_vel = self._build_joint_vel_obs_from_motion(joint_vel_motion)

        term_to_tensor = {
            "body_pos_b": body_pos_b.reshape(body_pos_w.shape[0], -1),
            "body_ori_b": body_ori_b,
            "body_lin_vel_b": body_lin_vel_b.reshape(body_pos_w.shape[0], -1),
            "body_ang_vel_b": body_ang_vel_b.reshape(body_pos_w.shape[0], -1),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        chunks: list[torch.Tensor] = []
        for name in self._amp_term_names:
            chunk = term_to_tensor[name]
            expected = self._amp_term_dims[name]
            actual = int(chunk.shape[-1])
            if actual != expected:
                raise ValueError(
                    f"AMP term '{name}' dim mismatch: expected={expected}, actual={actual}. "
                    "Check amp_obs config and expert-motion conversion."
                )
            chunks.append(chunk)
        return torch.cat(chunks, dim=-1)

    def _resolve_amp_term_spec(self) -> tuple[tuple[str, ...], dict[str, int]]:
        obs_mgr = getattr(self.env.unwrapped, "observation_manager", None)
        if obs_mgr is None or "amp_obs" not in obs_mgr.active_terms:
            raise KeyError("Observation group 'amp_obs' not found in observation_manager.")
        amp_term_names = tuple(obs_mgr.active_terms["amp_obs"])
        amp_term_dims = obs_mgr.group_obs_term_dim["amp_obs"]
        term_dim_map = {
            term_name: int(prod(dim_tuple)) for term_name, dim_tuple in zip(amp_term_names, amp_term_dims, strict=False)
        }
        supported = {
            "body_pos_b",
            "body_ori_b",
            "body_lin_vel_b",
            "body_ang_vel_b",
            "joint_pos",
            "joint_vel",
        }
        unsupported = [name for name in amp_term_names if name not in supported]
        if unsupported:
            raise KeyError(
                "Unsupported amp_obs term(s) for AMP expert sampling from motion: "
                + ", ".join(unsupported)
            )
        if not amp_term_names:
            raise ValueError("amp_obs has no terms configured.")
        return amp_term_names, term_dim_map

    def _build_joint_rel_obs_from_motion(self, motion_cmd, joint_pos_motion: torch.Tensor) -> torch.Tensor:
        expected_dim = self._amp_term_dims.get("joint_pos", joint_pos_motion.shape[1])
        motion_dim = int(joint_pos_motion.shape[1])
        out_dim = max(int(expected_dim), motion_dim)
        robot_ids = motion_cmd._motion_to_robot_joint_ids  # shape [27]
        default_joint_pos = motion_cmd.robot.data.default_joint_pos[:, robot_ids].to(joint_pos_motion.device)
        rel = joint_pos_motion - default_joint_pos

        out = torch.zeros(joint_pos_motion.shape[0], out_dim, device=joint_pos_motion.device, dtype=joint_pos_motion.dtype)
        out[:, :motion_dim] = rel
        return out[:, :expected_dim]

    def _build_joint_vel_obs_from_motion(self, joint_vel_motion: torch.Tensor) -> torch.Tensor:
        expected_dim = self._amp_term_dims.get("joint_vel", joint_vel_motion.shape[1])
        motion_dim = int(joint_vel_motion.shape[1])
        out_dim = max(int(expected_dim), motion_dim)

        out = torch.zeros(joint_vel_motion.shape[0], out_dim, device=joint_vel_motion.device, dtype=joint_vel_motion.dtype)
        
        out[:, :motion_dim] = joint_vel_motion
        return out[:, :expected_dim]
