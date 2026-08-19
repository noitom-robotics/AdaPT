"""Replay GMR ``.pkl`` in MuJoCo and record tracking ``.npz`` (same pipeline as ``csv_to_npz``).

Unlike ``gmr_pkl_to_mjlab_npz.py`` (direct pickle → arrays), this mirrors ``csv_to_npz.py``:
interpolate to ``output_fps``, write root + joints each step, ``sim.forward()``, then log
**full** ``joint_pos`` / ``joint_vel`` / ``body_*`` from ``robot.data``.

Output is written only to **local** ``.npz`` files (no Weights & Biases).

* ``--render`` saves an **offline ``.mp4``** next to each ``.npz`` (offscreen). It is **not**
  interactive Viser / live GUI; for that use ``play`` with ``--viewer viser``.

* ``--progress-mode`` ``auto`` (default): one **overall clip bar** when there are 2+ ``.pkl``;
  **per-frame** tqdm when there is only a single clip. Use ``clips`` / ``frames`` to force.

* ``--input-file`` may be a **single ``.pkl``** or a **directory** (recursive ``*.pkl``).
* If input is a file, ``--output-npz`` must be the target ``.npz`` path.
* If input is a directory, ``--output-npz`` must be an **output directory**; each clip is
  saved as ``<stem>.npz`` there.
* Optional pickle metadata ``grasp_frame_relative`` (input-frame index) is remapped to the
  replayed ``.npz`` timeline and written as ``grasp_frame_relative``.
* ``--robot`` selects the MuJoCo asset and replay joint order (``g1``, ``g1_grasp``, or ``atom_p3``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.motionio.gmr_pkl import iter_gmr_pkl_paths, load_gmr_pkl_dict
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

ReplayRobotKind = Literal["g1", "g1_grasp", "atom_p3"]


def _read_scalar_int_field(data: dict[str, Any], key: str) -> int | None:
  """Read a scalar int metadata field from a GMR pickle dict."""
  if key not in data:
    return None
  arr = np.asarray(data[key]).reshape(-1)
  if arr.size == 0:
    return None
  return int(arr[0])


def _map_input_frame_to_output(
  input_frame: int,
  input_frames: int,
  output_frames: int,
) -> int:
  """Map a frame index on the input timeline to the replayed output timeline."""
  if input_frames <= 1:
    return 0
  if output_frames <= 1:
    return 0
  phase = float(input_frame) / float(input_frames - 1)
  output_frame = int(round(phase * float(output_frames - 1)))
  return int(np.clip(output_frame, 0, output_frames - 1))


def resolve_replay_robot(
  robot: ReplayRobotKind,
  *,
  g1_dof_layout: Literal["29", "27"] = "27",
) -> tuple[SceneCfg, tuple[str, ...]]:
  """Scene (robot entity only) and ``dof_pos`` column / MJCF joint order for replay."""
  if robot == "g1":
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
    from mjlab.tasks.tracking.config.g1.replay_joint_names import get_g1_replay_joint_names

    scene_cfg = unitree_g1_flat_tracking_env_cfg().scene
    joint_names = tuple(get_g1_replay_joint_names(g1_dof_layout))
    return scene_cfg, joint_names

  if robot == "g1_grasp":
    raise ValueError(
      "robot='g1_grasp' is not included in AdaPT Tennis. Use robot='g1'."
    )

  if robot == "atom_p3":
    from mjlab.asset_zoo.robots.atom_p3.atom_constants import (
      ATOM_JOINT_NAMES,
      get_atom_robot_cfg,
    )

    scene_cfg = SceneCfg(entities={"robot": get_atom_robot_cfg()})
    return scene_cfg, tuple(ATOM_JOINT_NAMES)

  raise ValueError(f"Unknown robot {robot!r}; expected 'g1', 'g1_grasp', or 'atom_p3'.")


class GmrPklMotionLoader:
  """Load GMR pickle (``root_pos``, ``root_rot`` xyzw, ``dof_pos``) and resample like CSV loader."""

  def __init__(
    self,
    pkl_path: str,
    input_fps: float,
    output_fps: float,
    speed_scale: float,
    device: torch.device | str,
    joint_dof_indices: tuple[int, ...],
    frame_range: tuple[int, int] | None = None,
  ) -> None:
    self.pkl_path = str(Path(pkl_path).expanduser())
    self.input_fps = float(input_fps)
    self.output_fps = float(output_fps)
    self.speed_scale = float(speed_scale)
    if self.speed_scale <= 0.0:
      raise ValueError(f"speed_scale must be > 0, got {self.speed_scale}")
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.joint_dof_indices = joint_dof_indices
    self.frame_range = frame_range
    self.grasp_frame_relative: int | None = None
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()
    self._remap_grasp_frame_relative()

  def _load_motion(self) -> None:
    data = load_gmr_pkl_dict(self.pkl_path)
    if "root_pos" not in data or "root_rot" not in data or "dof_pos" not in data:
      raise KeyError(
        f"{self.pkl_path} must contain root_pos, root_rot (xyzw), dof_pos; got {list(data.keys())}"
      )
    root_pos = torch.as_tensor(
      np.asarray(data["root_pos"], dtype=np.float32), device=self.device
    )
    root_rot_xyzw = torch.as_tensor(
      np.asarray(data["root_rot"], dtype=np.float32), device=self.device
    )
    dof_full = torch.as_tensor(
      np.asarray(data["dof_pos"], dtype=np.float32), device=self.device
    )
    if root_rot_xyzw.shape[-1] != 4:
      raise ValueError(f"root_rot must be (T, 4) xyzw, got {root_rot_xyzw.shape}")
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    n_dofs = dof_full.shape[1]
    for j, col in enumerate(self.joint_dof_indices):
      if col < 0 or col >= n_dofs:
        raise IndexError(
          f"joint_dof_indices[{j}]={col} out of range for dof_pos width {n_dofs}"
        )
    dof_cols = torch.stack([dof_full[:, c] for c in self.joint_dof_indices], dim=1)

    grasp_frame_input = _read_scalar_int_field(data, "grasp_frame_relative")
    if self.frame_range is not None:
      a, b = self.frame_range
      root_pos = root_pos[a : b + 1]
      root_rot_wxyz = root_rot_wxyz[a : b + 1]
      dof_cols = dof_cols[a : b + 1]
      if grasp_frame_input is not None:
        grasp_frame_input -= int(a)

    # Normalize global heading: enforce first-frame root yaw == 0.
    # Apply the same fixed yaw correction to all subsequent frames.
    root_pos, root_rot_wxyz = self._normalize_initial_root_yaw(root_pos, root_rot_wxyz)

    self.motion_base_poss_input = root_pos
    self.motion_base_rots_input = root_rot_wxyz
    self.motion_dof_poss_input = dof_cols
    self._grasp_frame_input = grasp_frame_input

    self.input_frames = int(self.motion_base_poss_input.shape[0])
    if self.input_frames < 2:
      raise ValueError(
        f"Need at least 2 frames after slicing, got {self.input_frames} from {self.pkl_path}"
      )
    # Global time scaling for the whole clip:
    # - speed_scale < 1.0 => slow motion
    # - speed_scale > 1.0 => speed up
    self.duration = (self.input_frames - 1) * self.input_dt

  def _remap_grasp_frame_relative(self) -> None:
    """Map optional pickle ``grasp_frame_relative`` to replayed output frames."""
    if self._grasp_frame_input is None:
      self.grasp_frame_relative = None
      return
    max_input = max(self.input_frames - 1, 0)
    grasp_input = int(np.clip(int(self._grasp_frame_input), 0, max_input))
    self.grasp_frame_relative = _map_input_frame_to_output(
      grasp_input,
      self.input_frames,
      self.output_frames,
    )

  def _yaw_from_quat_wxyz(self, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Extract yaw (Z) from quaternion in wxyz layout."""
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)

  def _normalize_initial_root_yaw(
    self, root_pos: torch.Tensor, root_rot_wxyz: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate full clip so first-frame root yaw becomes zero."""
    yaw0 = self._yaw_from_quat_wxyz(root_rot_wxyz[0])
    half = -0.5 * yaw0
    qw = torch.cos(half)
    qz = torch.sin(half)
    q_bias = torch.tensor([qw, 0.0, 0.0, qz], device=self.device, dtype=root_rot_wxyz.dtype)
    q_bias = q_bias.unsqueeze(0).expand(root_rot_wxyz.shape[0], 4)

    # Orientation correction (world-frame yaw offset).
    root_rot_corr = quat_mul(q_bias, root_rot_wxyz)
    root_rot_corr = root_rot_corr / torch.linalg.norm(
      root_rot_corr, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)

    # Position correction: rotate trajectory around first-frame root position.
    c = torch.cos(-yaw0)
    s = torch.sin(-yaw0)
    center = root_pos[0:1].clone()
    rel = root_pos - center
    x_new = c * rel[:, 0] - s * rel[:, 1]
    y_new = s * rel[:, 0] + c * rel[:, 1]
    rel_corr = torch.stack([x_new, y_new, rel[:, 2]], dim=-1)
    root_pos_corr = rel_corr + center
    return root_pos_corr, root_rot_corr

  def _interpolate_motion(self) -> None:
    times = torch.arange(
      0, self.duration, self.output_dt * self.speed_scale, device=self.device, dtype=torch.float32
    )
    self.output_frames = int(times.shape[0])
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    print(
      f"GMR pkl interpolated: input_frames={self.input_frames}, input_fps={self.input_fps}, "
      f"output_frames={self.output_frames}, output_fps={self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(
      index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device)
    )
    blend = phase * (self.input_frames - 1) - index_0.float()
    return index_0, index_1, blend

  def _compute_velocities(self) -> None:
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names: tuple[str, ...],
  motion: GmrPklMotionLoader,
  output_fps: float,
  output_npz: str,
  render: bool,
  renderer: OffscreenRenderer | None = None,
  *,
  disable_frame_progress: bool = False,
) -> None:
  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(list(joint_names), preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": np.array([output_fps], dtype=np.float32),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  file_saved = False
  frames: list[Any] = []
  scene.reset()

  if not disable_frame_progress:
    print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render and not disable_frame_progress:
    print("Rendering enabled - generating video frames...")

  pbar = tqdm(
    total=motion.output_frames,
    desc="Frames",
    unit="frame",
    ncols=100,
    disable=disable_frame_progress,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)
      if not disable_frame_progress and frame_count % 100 == 0:
        pbar.set_description(f"Frames (t={frame_count / output_fps:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        if not disable_frame_progress:
          print("\nStacking arrays and saving data...")
        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ):
          log[k] = np.stack(log[k], axis=0)

        # Self-describing npz: persist the robot's own joint / body name ordering
        # so consumers (``MotionLoader``) can map by name when the runtime robot
        # XML has a different layout (extra finger / gripper bodies, swapped end
        # effectors, etc.).
        log["joint_names"] = np.array(list(robot.joint_names), dtype=object)
        log["body_names"] = np.array(list(robot.body_names), dtype=object)
        if motion.grasp_frame_relative is not None:
          log["grasp_frame_relative"] = np.array(
            [motion.grasp_frame_relative], dtype=np.int64
          )

        out_path = Path(output_npz).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **log)
        print(f"[INFO] Saved motion npz to: {out_path}")

        if render and frames:
          import mediapy as media

          video_path = out_path.with_suffix(".mp4")
          print(f"[INFO] Writing video to {video_path}")
          media.write_video(str(video_path), frames, fps=int(output_fps))


def _run_one_pkl(
  *,
  sim: Simulation,
  scene: Scene,
  renderer: OffscreenRenderer | None,
  joint_names: tuple[str, ...],
  indices: tuple[int, ...],
  pkl_path: Path,
  output_npz: Path,
  input_fps: float,
  output_fps: float,
  speed_scale: float,
  frame_range: tuple[int, int] | None,
  render: bool,
  disable_frame_progress: bool = False,
) -> None:

  motion = GmrPklMotionLoader(
    pkl_path=str(pkl_path),
    input_fps=input_fps,
    output_fps=output_fps,
    speed_scale=speed_scale,
    device=sim.device,
    joint_dof_indices=indices,
    frame_range=frame_range,
  )
  if disable_frame_progress:
    print(
      f"[INFO] {pkl_path.name}: {motion.output_frames} sim frames @ {output_fps} Hz -> {output_npz.name}"
    )
  run_sim(
    sim=sim,
    scene=scene,
    joint_names=joint_names,
    motion=motion,
    output_fps=output_fps,
    output_npz=str(output_npz),
    render=render,
    renderer=renderer,
    disable_frame_progress=disable_frame_progress,
  )


def main(
  input_file: str,
  output_npz: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  speed_scale: float = 1.0,
  device: str = "cuda:0",
  render: bool = False,
  progress_mode: Literal["auto", "frames", "clips"] = "auto",
  robot: ReplayRobotKind = "g1",
  g1_dof_layout: Literal["29", "27"] = "27",
  joint_dof_indices: tuple[int, ...] = (),
  frame_range: tuple[int, int] | None = None,
) -> None:
  """Convert GMR pickle(s) via sim replay. See module docstring for ``input_file`` / ``output_npz``.

  ``render``: offline ``.mp4`` only (offscreen), not live Viser.

  ``progress_mode``: ``auto`` uses a **single clip-level bar** when there are 2+ ``.pkl``,
  otherwise per-frame tqdm. ``clips`` / ``frames`` force that behavior.

  ``robot``: ``g1`` (Unitree G1), ``g1_grasp`` (G1 29-DOF + right gripper), or ``atom_p3``
  (Dobot Atom P3). ``g1_dof_layout`` applies only when ``robot=g1`` (``27`` / ``29`` are
  equivalent). ``joint_dof_indices``: one GMR
  ``dof_pos`` column per driven joint. If empty, uses ``0..N-1``. ``frame_range``:
  inclusive (first, last) frames.

  ``speed_scale``: global clip time scale. ``<1`` slows down the whole clip,
  ``>1`` speeds up (without changing input/output fps settings).
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA is not available. Falling back to CPU. This may be slow.")
    device = "cpu"

  scene_cfg, joint_names = resolve_replay_robot(robot, g1_dof_layout=g1_dof_layout)
  indices = joint_dof_indices
  if not indices:
    indices = tuple(range(len(joint_names)))

  in_path = Path(input_file).expanduser().resolve()
  out_arg = Path(output_npz).expanduser().resolve()

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  print(f"[INFO] robot={robot} n_replay_joints={len(joint_names)}")
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  if in_path.is_dir():
    out_arg.mkdir(parents=True, exist_ok=True)
    pkls = list(iter_gmr_pkl_paths(in_path, suffix=".pkl"))
    if not pkls:
      raise FileNotFoundError(
        f"No .pkl files under {in_path} (recursive). "
        "Put clips here or pass a single .pkl file as --input-file."
      )
    n_clips = len(pkls)
    if progress_mode == "auto":
      eff_progress = "clips" if n_clips > 1 else "frames"
    else:
      eff_progress = progress_mode
    print(f"[INFO] Found {n_clips} .pkl file(s); writing .npz under {out_arg}")

    if eff_progress == "clips":
      clip_bar = tqdm(
        pkls,
        total=n_clips,
        desc="Clips",
        unit="clip",
        ncols=115,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
      )
      for idx, pkl_path in enumerate(clip_bar, start=1):
        clip_bar.set_postfix_str(
          f"{pkl_path.name} | done {idx - 1}/{n_clips}, left {n_clips - idx}",
          refresh=True,
        )
        out_npz = out_arg / f"{pkl_path.stem}.npz"
        _run_one_pkl(
          sim=sim,
          scene=scene,
          renderer=renderer,
          joint_names=joint_names,
          indices=indices,
          pkl_path=pkl_path,
          output_npz=out_npz,
          input_fps=input_fps,
          output_fps=output_fps,
          speed_scale=speed_scale,
          frame_range=frame_range,
          render=render,
          disable_frame_progress=True,
        )
    else:
      for idx, pkl_path in enumerate(pkls, start=1):
        out_npz = out_arg / f"{pkl_path.stem}.npz"
        print(f"[INFO] ({idx}/{n_clips}) {pkl_path.name} -> {out_npz.name}")
        _run_one_pkl(
          sim=sim,
          scene=scene,
          renderer=renderer,
          joint_names=joint_names,
          indices=indices,
          pkl_path=pkl_path,
          output_npz=out_npz,
          input_fps=input_fps,
          output_fps=output_fps,
          speed_scale=speed_scale,
          frame_range=frame_range,
          render=render,
          disable_frame_progress=False,
        )
  else:
    if not in_path.is_file():
      raise FileNotFoundError(f"Input path is not a file or directory: {in_path}")
    if out_arg.suffix.lower() != ".npz":
      raise ValueError(
        "When --input-file is a single .pkl file, --output-npz must end with .npz "
        f"(got {out_arg})"
      )
    out_arg.parent.mkdir(parents=True, exist_ok=True)
    n_one = 1
    eff_progress = progress_mode if progress_mode != "auto" else "frames"
    if eff_progress == "clips":
      clip_bar = tqdm(
        [in_path],
        total=n_one,
        desc="Clips",
        unit="clip",
        ncols=115,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
      )
      for idx, pkl_path in enumerate(clip_bar, start=1):
        clip_bar.set_postfix_str(
          f"{pkl_path.name} | done {idx - 1}/{n_one}, left {n_one - idx}", refresh=True
        )
        _run_one_pkl(
          sim=sim,
          scene=scene,
          renderer=renderer,
          joint_names=joint_names,
          indices=indices,
          pkl_path=pkl_path,
          output_npz=out_arg,
          input_fps=input_fps,
          output_fps=output_fps,
          speed_scale=speed_scale,
          frame_range=frame_range,
          render=render,
          disable_frame_progress=True,
        )
    else:
      _run_one_pkl(
        sim=sim,
        scene=scene,
        renderer=renderer,
        joint_names=joint_names,
        indices=indices,
        pkl_path=in_path,
        output_npz=out_arg,
        input_fps=input_fps,
        output_fps=output_fps,
        speed_scale=speed_scale,
        frame_range=frame_range,
        render=render,
        disable_frame_progress=False,
      )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
