"""Export tennis hierarchical ONNX bundle (high_level/mvae/tracker)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import tyro
import mjlab

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tennis.rl import TennisOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends


@dataclass
class ExportTennisOnnxBundleCfg:
  checkpoint_file: Path
  output_dir: Path | None = None
  device: str = "cuda:0"


def run_export(task_id: str, cfg: ExportTennisOnnxBundleCfg) -> None:
  configure_torch_backends()
  ckpt = cfg.checkpoint_file.expanduser().resolve()
  if not ckpt.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  rl_cfg = load_rl_cfg(task_id)
  agent_cfg = asdict(rl_cfg)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    raise RuntimeError(f"Task {task_id} has no custom runner class; expected TennisOnPolicyRunner.")

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)
  try:
    runner = runner_cls(vec_env, agent_cfg, log_dir=None, device=cfg.device)
    if not isinstance(runner, TennisOnPolicyRunner):
      raise TypeError(f"Runner for {task_id} is {type(runner).__name__}, not TennisOnPolicyRunner")
    runner.load(str(ckpt), map_location=cfg.device)

    out_dir = cfg.output_dir.expanduser().resolve() if cfg.output_dir else ckpt.parent / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    high = runner._export_high_level_onnx(out_dir)
    mvae = runner._export_mvae_onnx(out_dir)
    tracker = runner._materialize_tracker_onnx(out_dir)

    print(f"[Export] high_level: {high}")
    print(f"[Export] mvae: {mvae if mvae is not None else '(skip/failed)'}")
    print(f"[Export] tracker: {tracker if tracker is not None else '(skip/failed)'}")
  finally:
    vec_env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    ExportTennisOnnxBundleCfg,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
