"""Headless play recording: save an ``.mp4`` without debug_vis overlays.

Example::

  uv run record-play-video \\
    --task-id Mjlab-AdaPT-Tennis-Flat-Unitree-G1 \\
    --checkpoint-file /path/to/model.pt \\
    --motion-file /path/to/clip.npz
"""

from __future__ import annotations

import os

# Headless servers have no DISPLAY; GLFW cannot create a GL context.
# EGL (same as train.py) is required before mujoco.Renderer is constructed.
os.environ.setdefault("MUJOCO_GL", "egl")

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from mjlab.scripts.play import PlayConfig, run_play

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_TASK = "Mjlab-AdaPT-Tennis-Flat-Unitree-G1"


@dataclass(frozen=True)
class RecordPlayVideoConfig:
  task_id: str = _DEFAULT_TASK
  checkpoint_file: str = ""
  motion_file: str = ""
  motion_warmup_steps: int | None = None
  motion_warmup_step_source: Literal["env", "learning_iteration", "episode"] | None = None
  duration_s: float = 8.0
  """Wall-clock simulation time to record (converted to env steps)."""
  video_height: int = 720
  video_width: int = 1280
  num_envs: int = 1
  device: str | None = None
  video_folder: str | None = None
  """Directory for the raw ``rl-video-step-0.mp4``. Default: ``logs/play_videos``."""
  output: str | None = None
  """Optional path to copy the finished ``.mp4`` (parent dirs are created)."""
  racket_hand: Literal["left", "right"] | None = None
  """G1 tennis: ``left`` / ``right`` racket XML. Unset keeps the task default (left)."""


def _latest_mp4(folder: Path) -> Path | None:
  videos = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
  return videos[-1] if videos else None


def main() -> None:
  cfg = tyro.cli(RecordPlayVideoConfig, config=mjlab.TYRO_FLAGS)
  video_folder = (
    Path(cfg.video_folder).expanduser().resolve()
    if cfg.video_folder
    else (_PROJECT_ROOT / "logs" / "play_videos").resolve()
  )
  video_folder.mkdir(parents=True, exist_ok=True)

  play_cfg = PlayConfig(
    checkpoint_file=cfg.checkpoint_file,
    motion_file=cfg.motion_file,
    motion_warmup_steps=cfg.motion_warmup_steps,
    motion_warmup_step_source=cfg.motion_warmup_step_source,
    num_envs=cfg.num_envs,
    device=cfg.device,
    video=True,
    video_headless=True,
    debug_vis=False,
    video_duration_s=cfg.duration_s,
    video_height=cfg.video_height,
    video_width=cfg.video_width,
    video_folder=str(video_folder),
    racket_hand=cfg.racket_hand,
  )
  print(
    "[INFO] Recording play video (debug_vis=off, headless)\n"
    f"  task={cfg.task_id}\n"
    f"  checkpoint={cfg.checkpoint_file}\n"
    f"  motion={cfg.motion_file}\n"
    f"  duration_s={cfg.duration_s}  resolution={cfg.video_width}x{cfg.video_height}"
  )
  run_play(cfg.task_id, play_cfg)

  recorded = _latest_mp4(video_folder)
  if recorded is None:
    print(f"[ERROR] No .mp4 written under {video_folder}", file=sys.stderr)
    sys.exit(1)
  print(f"[INFO] Recorded video: {recorded}")
  if cfg.output:
    dest = Path(cfg.output).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(recorded, dest)
    print(f"[INFO] Copied to: {dest}")


if __name__ == "__main__":
  main()
