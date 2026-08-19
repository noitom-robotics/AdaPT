"""Batch-convert GMR ``.pkl`` clips to mjlab tracking ``.npz`` files.

Example::

  uv run python -m mjlab.scripts.gmr_pkl_to_mjlab_npz \\
    --input-folder /data/gmr_out \\
    --output-dir /data/mjlab_motions \\
    --body-names pelvis torso_link \\
    --joint-dof-indices 0 1 2 \\
    --task-id Mjlab-Tracking-Flat-Unitree-G1

If ``--task-id`` is set, ``--body-names`` can be omitted (taken from the task motion command).
``--joint-dof-indices`` must list one GMR ``dof_pos`` column index per mjlab joint row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

import mjlab
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.motionio.gmr_pkl import (
  convert_gmr_pkl_to_mjlab_npz_arrays,
  iter_gmr_pkl_paths,
  load_gmr_pkl_dict,
)


@dataclass(frozen=True)
class CliConfig:
  input_folder: str
  output_dir: str
  """Directory to write one ``.npz`` per input ``.pkl`` (same basename)."""

  body_names: tuple[str, ...] = ()
  """Must match ``MotionCommandCfg.body_names`` order for training."""

  task_id: str | None = None
  """If set, load ``body_names`` from this registry task (overrides ``body_names``)."""

  joint_dof_indices: tuple[int, ...] = ()
  """Column indices into GMR ``dof_pos`` for each mjlab ``joint_pos`` row."""

  drop_last_frame: bool = False
  fps: float | None = None
  suffix: str = ".pkl"


def _body_names_from_task(task_id: str) -> tuple[str, ...]:
  from mjlab.tasks.registry import load_env_cfg

  env_cfg = load_env_cfg(task_id, play=False)
  cmd = env_cfg.commands.get("motion")
  if not isinstance(cmd, MotionCommandCfg):
    raise ValueError(f"Task {task_id!r} has no MotionCommandCfg motion command.")
  if not cmd.body_names:
    raise ValueError(f"Task {task_id!r} has empty motion.body_names.")
  return tuple(cmd.body_names)


def main(cfg: CliConfig) -> None:
  out_dir = Path(cfg.output_dir).expanduser().resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  body_names = cfg.body_names
  if cfg.task_id is not None:
    body_names = _body_names_from_task(cfg.task_id)
  if not body_names:
    raise ValueError("Provide --body-names or --task-id with non-empty body_names.")

  if not cfg.joint_dof_indices:
    raise ValueError("--joint-dof-indices is required (one int per output joint dimension).")

  paths = list(iter_gmr_pkl_paths(cfg.input_folder, suffix=cfg.suffix))
  if not paths:
    raise FileNotFoundError(f"No *{cfg.suffix} files under {cfg.input_folder!r}")

  print(f"[INFO] Found {len(paths)} files. Writing to {out_dir}")
  for pkl_path in paths:
    try:
      raw = load_gmr_pkl_dict(pkl_path)
      arrays = convert_gmr_pkl_to_mjlab_npz_arrays(
        raw,
        body_names=body_names,
        joint_dof_indices=cfg.joint_dof_indices,
        drop_last_frame=cfg.drop_last_frame,
        fps_override=cfg.fps,
      )
      out_path = out_dir / (pkl_path.stem + ".npz")
      np.savez(out_path, **arrays)
      print(f"  OK {pkl_path.name} -> {out_path.name} T={arrays['joint_pos'].shape[0]}")
    except Exception as e:
      print(f"  FAIL {pkl_path}: {e}")


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401 — register tasks for --task-id

  tyro.cli(main, config=mjlab.TYRO_FLAGS)
