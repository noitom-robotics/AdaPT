import os
import json
from pathlib import Path

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name
    
  def _export_sim2sim_info(self, export_dir: Path) -> Path | None:
    """Export sim2sim-compatible info.json alongside JIT bundle."""
    try:
      env = self.env.unwrapped
      robot = env.scene["robot"]

      # Actuated joint names in natural joint order.
      _, dof_names = robot.find_joints_by_actuator_names((".*",))
      dof_ids, _ = robot.find_joints(dof_names, preserve_order=True)

      # Resolve actuator IDs for per-joint stiffness/damping/torque limits.
      joint_name_to_ctrl_id: dict[str, int] = {}
      for actuator in robot.spec.actuators:
        joint_name = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_name] = actuator.id
      ctrl_ids = [
        joint_name_to_ctrl_id[n] for n in dof_names if n in joint_name_to_ctrl_id
      ]

      mj = env.sim.mj_model
      stiffness = mj.actuator_gainprm[ctrl_ids, 0].tolist()
      damping = (-mj.actuator_biasprm[ctrl_ids, 2]).tolist()
      torque_limits = mj.actuator_forcerange[ctrl_ids, 1].tolist()

      default_joint_pos = robot.data.default_joint_pos[0, dof_ids].cpu().tolist()

      # Tracking motion command/body keyframes order.
      motion_cmd = env.command_manager.get_term("motion")
      keyframe_names = list(getattr(motion_cmd.cfg, "body_names", ()))

      data = {
        "DOF NAMES": list(dof_names),
        "KEYFRAME NAMES": keyframe_names,
        "DEFAULT JOINT ANGLES": default_joint_pos,
        "STIFFNESS": {
          name: float(v) for name, v in zip(dof_names, stiffness, strict=False)
        },
        "DAMPING": {
          name: float(v) for name, v in zip(dof_names, damping, strict=False)
        },
        "TORQUE LIMITS": [float(v) for v in torque_limits],
      }

      out_path = export_dir / "info.json"
      out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
      )
      return out_path
    except Exception as e:
      print(f"[WARN] sim2sim info export failed: {e}")
      return None
    
  def export_policy_to_jit(
    self, path: str, filename: str = "policy.pt"
  ) -> None:
    os.makedirs(path, exist_ok=True)
    jit_model = self.alg.get_policy().as_jit()
    jit_model.to("cpu")
    jit_model.eval()
    traced_script_module = torch.jit.script(jit_model)
    traced_script_module.save(os.path.join(path, filename))

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir = self._get_export_paths(path)[0]
    # import ipdb
    # ipdb.set_trace()
    jit_dir = policy_dir / "jit"
    jit_filename = f"modeljit_{self.current_learning_iteration}.pt"
    try:
      self.export_policy_to_jit(str(jit_dir), jit_filename)
      info_path = self._export_sim2sim_info(jit_dir)
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        jit_path = jit_dir / jit_filename
        wandb.save(str(jit_path), base_path=str(policy_dir))
        if info_path is not None:
          wandb.save(str(info_path), base_path=str(policy_dir))
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore
          self.registry_name = None
    except Exception as e:
      print(f"[WARN] JIT export failed (training continues): {e}")
