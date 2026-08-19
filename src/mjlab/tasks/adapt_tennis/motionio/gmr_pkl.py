"""Load GMR-style retargeted ``.pkl`` and convert to mjlab tracking ``.npz`` layout.

Expected GMR keys (per clip, as in legged_gym ``motionlib_gmr``):

  - ``root_pos`` (T, 3), ``root_rot`` (T, 4) xyzw — optional if link data covers root body
  - ``dof_pos`` (T, N_dof)
  - ``link_position`` (T, N_link, 3)
  - ``link_orientation`` (T, N_link, 3) euler XYZ (radians) **or** (T, N_link, 4) quat
  - ``link_velocity`` (T, N_link, 3), ``link_angular_velocity`` (T, N_link, 3)
  - ``link_body_list``: list[str] aligned with link index

mjlab ``MotionLoader`` expects ``.npz`` with:

  - ``joint_pos`` (T, num_robot_joints), ``joint_vel`` (T, num_robot_joints)
  - ``body_pos_w``, ``body_lin_vel_w``, ``body_ang_vel_w`` (T, num_tracked_bodies, 3)
  - ``body_quat_w`` (T, num_tracked_bodies, 4) **wxyz** (Isaac / mjlab convention)
  - optional ``fps`` (scalar) for metadata (not read by ``MotionLoader`` today)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from mjlab.utils.lab_api.math import convert_quat, quat_from_euler_xyz


class CompatUnpickler(pickle.Unpickler):
  """Pickle loader tolerant of numpy 1.x / 2.x ``numpy.core`` vs ``numpy._core`` paths."""

  def find_class(self, module: str, name: str) -> Any:
    if module.startswith("numpy.core"):
      new_module = module.replace("numpy.core", "numpy._core")
      try:
        return super().find_class(new_module, name)
      except (AttributeError, ModuleNotFoundError):
        try:
          return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
          pass
    elif module.startswith("numpy._core"):
      old_module = module.replace("numpy._core", "numpy.core")
      try:
        return super().find_class(old_module, name)
      except (AttributeError, ModuleNotFoundError):
        try:
          return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
          pass
    if "multiarray" in module or "_multiarray" in module:
      alts: list[str] = []
      if "numpy.core" in module:
        alts = [
          module.replace("numpy.core", "numpy._core"),
          module,
          "numpy._core.multiarray",
          "numpy.core.multiarray",
        ]
      elif "numpy._core" in module:
        alts = [
          module.replace("numpy._core", "numpy.core"),
          module,
          "numpy.core.multiarray",
          "numpy._core.multiarray",
        ]
      else:
        alts = [
          "numpy._core.multiarray",
          "numpy.core.multiarray",
          "numpy._core._multiarray_umath",
          "numpy.core._multiarray_umath",
        ]
      for alt in alts:
        try:
          return super().find_class(alt, name)
        except (AttributeError, ModuleNotFoundError):
          continue
    return super().find_class(module, name)


def load_gmr_pkl_dict(path: str | Path) -> dict[str, Any]:
  """Load one GMR ``.pkl`` into a string-keyed dict (numpy arrays where applicable)."""
  path = Path(path)
  with path.open("rb") as f:
    try:
      data = pickle.load(f)
    except (ModuleNotFoundError, AttributeError) as e:
      if "numpy" in str(e) or "_core" in str(e):
        f.seek(0)
        data = CompatUnpickler(f).load()
      else:
        raise
  if not isinstance(data, dict):
    raise TypeError(f"Expected dict in {path}, got {type(data)}")
  return data


def iter_gmr_pkl_paths(folder: str | Path, suffix: str = ".pkl") -> Iterator[Path]:
  """Yield sorted ``.pkl`` paths under ``folder`` (recursive)."""
  folder = Path(folder).expanduser().resolve()
  if not folder.is_dir():
    return
  paths: list[Path] = []
  for root, _, files in os.walk(folder):
    root_p = Path(root)
    for name in files:
      if name.endswith(suffix):
        paths.append(root_p / name)
  paths.sort()
  yield from paths


def _euler_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
  """euler (T, 3) roll, pitch, yaw -> quat (T, 4) wxyz."""
  t = torch.as_tensor(euler, dtype=torch.float32)
  r, p, y = t[:, 0], t[:, 1], t[:, 2]
  q = quat_from_euler_xyz(r, p, y)
  return q.cpu().numpy().astype(np.float32)


def _orientation_to_quat_wxyz(ori: np.ndarray) -> np.ndarray:
  """(T, 3) euler XYZ or (T, 4) quaternion (xyzw or wxyz heuristics)."""
  if ori.shape[-1] == 3:
    return _euler_to_quat_wxyz(ori)
  if ori.shape[-1] == 4:
    t = torch.as_tensor(ori, dtype=torch.float32)
    mean_abs = t.abs().mean(dim=0)
    w_first = mean_abs[0] >= mean_abs[1:].max()
    if w_first:
      return t.cpu().numpy().astype(np.float32)
    q_xyzw = convert_quat(t, to="wxyz")
    return q_xyzw.cpu().numpy().astype(np.float32)
  raise ValueError(f"link_orientation last dim must be 3 or 4, got {ori.shape}")


def convert_gmr_pkl_to_mjlab_npz_arrays(
  data: dict[str, Any],
  body_names: tuple[str, ...],
  joint_dof_indices: tuple[int, ...],
  *,
  drop_last_frame: bool = False,
  fps_override: float | None = None,
) -> dict[str, np.ndarray]:
  """Build mjlab motion arrays from one GMR clip dict."""
  required = ("dof_pos", "link_position", "link_body_list")
  for k in required:
    if k not in data:
      raise KeyError(f"GMR data missing key {k!r}")

  dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
  link_pos = np.asarray(data["link_position"], dtype=np.float32)
  link_body_list = [str(x) for x in data["link_body_list"]]

  if dof_pos.ndim != 2:
    raise ValueError(f"dof_pos must be (T, N), got {dof_pos.shape}")
  if link_pos.ndim != 3:
    raise ValueError(f"link_position must be (T, L, 3), got {link_pos.shape}")

  t_frames = dof_pos.shape[0]
  if link_pos.shape[0] != t_frames:
    raise ValueError(f"dof_pos T={t_frames} != link_position T={link_pos.shape[0]}")

  if drop_last_frame and t_frames > 1:
    sl = slice(None, -1)
    dof_pos = dof_pos[sl]
    link_pos = link_pos[sl]
    t_frames -= 1
  else:
    sl = slice(None)

  for ji, col in enumerate(joint_dof_indices):
    if col < 0 or col >= dof_pos.shape[1]:
      raise IndexError(
        f"joint_dof_indices[{ji}]={col} out of range for dof width {dof_pos.shape[1]}"
      )

  joint_pos = np.stack([dof_pos[:, col] for col in joint_dof_indices], axis=1).astype(
    np.float32
  )

  fps = float(fps_override if fps_override is not None else data.get("fps", 30.0))
  if joint_pos.shape[0] > 1:
    joint_vel = (np.gradient(joint_pos, axis=0) * fps).astype(np.float32)
  else:
    joint_vel = np.zeros_like(joint_pos)

  name_to_li = {name: i for i, name in enumerate(link_body_list)}
  body_idx: list[int] = []
  for name in body_names:
    if name not in name_to_li:
      raise KeyError(
        f"body_name {name!r} not in link_body_list (have {len(link_body_list)} links)"
      )
    body_idx.append(name_to_li[name])
  n_b = len(body_idx)

  body_pos_w = np.zeros((t_frames, n_b, 3), dtype=np.float32)
  body_lin_vel_w = np.zeros((t_frames, n_b, 3), dtype=np.float32)
  body_ang_vel_w = np.zeros((t_frames, n_b, 3), dtype=np.float32)
  body_quat_w = np.zeros((t_frames, n_b, 4), dtype=np.float32)
  body_quat_w[..., 0] = 1.0

  body_pos_w[:] = link_pos[:, body_idx, :]

  if "link_velocity" in data:
    lv = np.asarray(data["link_velocity"], dtype=np.float32)[sl]
    if lv.shape[:2] == (t_frames, len(link_body_list)):
      body_lin_vel_w[:] = lv[:, body_idx, :]

  if "link_angular_velocity" in data:
    av = np.asarray(data["link_angular_velocity"], dtype=np.float32)[sl]
    if av.shape[:2] == (t_frames, len(link_body_list)):
      body_ang_vel_w[:] = av[:, body_idx, :]

  if "link_orientation" in data:
    lo = np.asarray(data["link_orientation"], dtype=np.float32)[sl]
    if lo.shape[:2] != (t_frames, len(link_body_list)):
      raise ValueError(
        f"link_orientation shape {lo.shape} inconsistent with T={t_frames}, L={len(link_body_list)}"
      )
    for bi, li in enumerate(body_idx):
      body_quat_w[:, bi, :] = _orientation_to_quat_wxyz(lo[:, li, :])
  else:
    raise KeyError("GMR data missing link_orientation (needed for body_quat_w)")

  return {
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
    "fps": np.array(fps, dtype=np.float32),
  }
