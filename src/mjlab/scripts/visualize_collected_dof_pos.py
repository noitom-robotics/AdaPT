"""Visualize ``dof_pos`` rollouts collected by ``play --collect-dof-pos-n``.

Collected ``.pkl`` files store ``dof_pos`` and, when present, per-frame
``root_pos`` / ``root_quat`` (wxyz, world frame). Older pickles without root
fields fall back to a fixed default / CLI root pose.

Example:
  uv run viz-collected-dof logs/play_dof_pos/20260101_120000_Mjlab-ServeTracking-Flat-Unitree-G1-Stage2-Gripper-Dt
  uv run viz-collected-dof logs/play_dof_pos/.../rollout_000.pkl --robot g1_left
"""

from __future__ import annotations

import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import torch
import tyro
import viser

import mjlab
from mjlab.asset_zoo.robots.atom_p3.atom_constants import get_atom_robot_cfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.viser import ViserMujocoScene

RobotKind = Literal["g1", "g1_left", "g1_right", "atom_p3"]


@dataclass(frozen=True)
class RolloutData:
  """One collected rollout from ``play`` DOF collection."""

  path: Path
  dof_pos: np.ndarray
  motion_time: np.ndarray
  sim_time: np.ndarray
  root_pos: np.ndarray | None
  root_quat: np.ndarray | None
  task_id: str | None
  rollout_index: int | None


def _resolve_robot_cfg(robot: RobotKind) -> EntityCfg:
  if robot == "g1":
    return get_g1_robot_cfg()
  if robot == "g1_left":
    return get_g1_robot_cfg(racket_hand="left")
  if robot == "g1_right":
    return get_g1_robot_cfg(racket_hand="right")
  if robot == "atom_p3":
    return get_atom_robot_cfg()
  raise ValueError(f"Unknown robot {robot!r}")


def _load_rollout_pkl(path: Path) -> RolloutData:
  with path.open("rb") as f:
    payload: dict[str, Any] = pickle.load(f)
  dof_pos = np.asarray(payload["dof_pos"], dtype=np.float32)
  if dof_pos.ndim != 2:
    raise ValueError(f"{path}: dof_pos must be (T, num_joints), got {dof_pos.shape}")
  motion_time = np.asarray(payload.get("motion_time", []), dtype=np.float32)
  sim_time = np.asarray(payload.get("sim_time", []), dtype=np.float32)
  root_pos_raw = payload.get("root_pos")
  root_quat_raw = payload.get("root_quat")
  root_pos = (
    np.asarray(root_pos_raw, dtype=np.float32)
    if root_pos_raw is not None and len(root_pos_raw) > 0
    else None
  )
  root_quat = (
    np.asarray(root_quat_raw, dtype=np.float32)
    if root_quat_raw is not None and len(root_quat_raw) > 0
    else None
  )
  if root_pos is not None and root_pos.shape[0] != dof_pos.shape[0]:
    raise ValueError(
      f"{path}: root_pos length {root_pos.shape[0]} != dof_pos frames {dof_pos.shape[0]}"
    )
  if root_quat is not None and root_quat.shape[0] != dof_pos.shape[0]:
    raise ValueError(
      f"{path}: root_quat length {root_quat.shape[0]} != dof_pos frames {dof_pos.shape[0]}"
    )
  return RolloutData(
    path=path,
    dof_pos=dof_pos,
    motion_time=motion_time,
    sim_time=sim_time,
    root_pos=root_pos,
    root_quat=root_quat,
    task_id=payload.get("task_id"),
    rollout_index=payload.get("rollout_index"),
  )


def _discover_rollouts(input_path: Path) -> list[RolloutData]:
  if input_path.is_file():
    return [_load_rollout_pkl(input_path)]
  if not input_path.is_dir():
    raise FileNotFoundError(f"Input not found: {input_path}")
  pkls = sorted(input_path.glob("rollout_*.pkl"))
  if not pkls:
    pkls = sorted(input_path.glob("*.pkl"))
  if not pkls:
    raise FileNotFoundError(f"No .pkl rollouts under {input_path}")
  return [_load_rollout_pkl(p) for p in pkls]


@dataclass
class VizCollectedDofPosCfg:
  """Viser viewer for collected ``dof_pos`` trajectories."""

  input_path: str
  """``.pkl`` file or directory containing ``rollout_*.pkl`` from play collection."""

  robot: RobotKind = "g1"
  """Robot asset (must match the model used during collection)."""

  root_pos: tuple[float, float, float] | None = None
  """Fixed root XYZ (world). ``None`` uses the robot default init pose."""

  root_quat_wxyz: tuple[float, float, float, float] | None = None
  """Fixed root quaternion wxyz. ``None`` uses the robot default init pose."""

  port: int | None = None
  """Viser server port."""

  playback_fps: float = 30.0
  """Frames per second when playback is enabled."""


class CollectedDofPosViewer:
  """Interactive Viser viewer for fixed-root joint pose playback."""

  def __init__(self, cfg: VizCollectedDofPosCfg) -> None:
    self.cfg = cfg
    self.rollouts = _discover_rollouts(Path(cfg.input_path).expanduser().resolve())
    if not self.rollouts:
      raise RuntimeError("No rollouts loaded.")

    scene_cfg = SceneCfg(num_envs=1, entities={"robot": _resolve_robot_cfg(cfg.robot)})
    device = "cpu"
    self.scene = Scene(scene_cfg, device=device)
    model = self.scene.compile()
    sim_cfg = SimulationCfg()
    self.sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
    self.scene.reset()
    self.robot = self.scene["robot"]
    self._viz_data = mujoco.MjData(self.sim.mj_model)

    n_joints = int(self.robot.data.joint_pos.shape[1])
    for i, rollout in enumerate(self.rollouts):
      if rollout.dof_pos.shape[1] != n_joints:
        raise ValueError(
          f"{rollout.path}: dof_pos dim {rollout.dof_pos.shape[1]} != robot "
          f"joint count {n_joints} (robot={cfg.robot})."
        )

    self._root_state = self._build_fixed_root_state()
    self.server = viser.ViserServer(port=cfg.port, label="Collected DOF viewer")
    self.viser_scene = ViserMujocoScene(self.server, self.sim.mj_model, num_envs=1)

    self.rollout_idx = 0
    self.frame_idx = 0
    self.playing = False

  def _build_fixed_root_state(self) -> torch.Tensor:
    root = self.robot.data.default_root_state.clone()
    if self.cfg.root_pos is not None:
      root[:, 0:3] = torch.tensor(self.cfg.root_pos, dtype=root.dtype, device=root.device)
    if self.cfg.root_quat_wxyz is not None:
      root[:, 3:7] = torch.tensor(
        self.cfg.root_quat_wxyz, dtype=root.dtype, device=root.device
      )
    root[:, 7:] = 0.0
    return root

  @property
  def _active(self) -> RolloutData:
    return self.rollouts[self.rollout_idx]

  def _apply_frame(self, frame_idx: int) -> None:
    rollout = self._active
    frame_idx = int(np.clip(frame_idx, 0, rollout.dof_pos.shape[0] - 1))
    self.frame_idx = frame_idx

    root_state = self._root_state.clone()
    if rollout.root_pos is not None:
      root_state[0, 0:3] = torch.from_numpy(rollout.root_pos[frame_idx]).to(
        root_state.device, dtype=root_state.dtype
      )
    if rollout.root_quat is not None:
      root_state[0, 3:7] = torch.from_numpy(rollout.root_quat[frame_idx]).to(
        root_state.device, dtype=root_state.dtype
      )
    self.robot.write_root_state_to_sim(root_state)
    joint_pos = self.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    joint_pos[0] = torch.from_numpy(rollout.dof_pos[frame_idx]).to(joint_pos.device)
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    self.sim.forward()
    self._viz_data.qpos[:] = self.sim.data.qpos[0].detach().cpu().numpy()
    self._viz_data.qvel[:] = 0.0
    mujoco.mj_forward(self.sim.mj_model, self._viz_data)
    self.viser_scene.update_from_mjdata(self._viz_data)

  def _info_html(self) -> str:
    rollout = self._active
    n_frames = rollout.dof_pos.shape[0]
    mt = (
      float(rollout.motion_time[self.frame_idx])
      if rollout.motion_time.shape[0] > self.frame_idx
      else float("nan")
    )
    st = (
      float(rollout.sim_time[self.frame_idx])
      if rollout.sim_time.shape[0] > self.frame_idx
      else float("nan")
    )
    return f"""
      <div style="font-size: 0.85em; line-height: 1.25; padding: 0 1em 0.5em 1em;">
      <strong>Rollout:</strong> {self.rollout_idx} / {len(self.rollouts) - 1}
      ({rollout.path.name})<br/>
      <strong>Frame:</strong> {self.frame_idx} / {max(n_frames - 1, 0)}<br/>
      <strong>motion_time:</strong> {mt:.4f} s<br/>
      <strong>sim_time:</strong> {st:.4f} s<br/>
      <strong>robot:</strong> {self.cfg.robot}<br/>
      <strong>joints:</strong> {rollout.dof_pos.shape[1]}
      </div>
    """

  def setup(self) -> None:
    self.info_html = self.server.gui.add_html(self._info_html())
    self.viser_scene.create_scene_gui(show_debug_viz_control=False)

    with self.server.gui.add_folder("Rollout"):
      self.rollout_slider = self.server.gui.add_slider(
        "Index",
        min=0,
        max=max(len(self.rollouts) - 1, 0),
        step=1,
        initial_value=0,
      )

      @self.rollout_slider.on_update
      def _(_) -> None:
        self.rollout_idx = int(self.rollout_slider.value)
        self.frame_idx = 0
        if hasattr(self, "frame_slider"):
          self.frame_slider.max = max(self._active.dof_pos.shape[0] - 1, 0)
          self.frame_slider.value = 0
        self._apply_frame(0)
        self.info_html.content = self._info_html()

    with self.server.gui.add_folder("Playback"):
      n0 = max(self._active.dof_pos.shape[0] - 1, 0)
      self.frame_slider = self.server.gui.add_slider(
        "Frame",
        min=0,
        max=n0,
        step=1,
        initial_value=0,
      )

      @self.frame_slider.on_update
      def _(_) -> None:
        if not self.playing:
          self._apply_frame(int(self.frame_slider.value))
          self.info_html.content = self._info_html()

      self.play_btn = self.server.gui.add_button("Play / Pause")

      @self.play_btn.on_click
      def _(_) -> None:
        self.playing = not self.playing

    self._apply_frame(0)

  def run(self) -> None:
    print(f"[INFO] Loaded {len(self.rollouts)} rollout(s) from {self.cfg.input_path}")
    print("[INFO] Root pose is fixed; only joint angles change. Ctrl+C to exit.")
    dt = 1.0 / max(float(self.cfg.playback_fps), 1e-6)
    try:
      while True:
        if self.viser_scene.needs_update:
          self.viser_scene.refresh_visualization()
        if self.playing and self._active.dof_pos.shape[0] > 1:
          next_frame = (self.frame_idx + 1) % self._active.dof_pos.shape[0]
          self.frame_slider.value = next_frame
          self._apply_frame(next_frame)
          self.info_html.content = self._info_html()
          time.sleep(dt)
        else:
          time.sleep(0.05)
    except KeyboardInterrupt:
      print("\n[INFO] Shutting down...")
      self.server.stop()


def _normalize_cli_args(argv: list[str]) -> list[str]:
  """Map ``--input_path`` / bare ``PATH`` to tyro's ``--input-path``."""
  out = [a.replace("--input_path", "--input-path") for a in argv]
  if not out or out[0] in ("-h", "--help"):
    return out
  has_input = any(
    a == "--input-path" or a.startswith("--input-path=") for a in out
  )
  if not has_input and not out[0].startswith("-"):
    out = ["--input-path", out[0], *out[1:]]
  return out


def main() -> None:
  cfg = tyro.cli(
    VizCollectedDofPosCfg,
    args=_normalize_cli_args(sys.argv[1:]),
    description=__doc__,
    config=mjlab.TYRO_FLAGS,
  )
  viewer = CollectedDofPosViewer(cfg)
  viewer.setup()
  viewer.run()


if __name__ == "__main__":
  main()
