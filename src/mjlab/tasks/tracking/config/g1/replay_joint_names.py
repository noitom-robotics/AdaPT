"""Joint name order for ``csv_to_npz`` / ``gmr_pkl_sim_to_npz`` replay (G1 tracking).

Must match the order you drive with CSV / GMR ``dof_pos`` columns via
``joint_dof_indices``, and every name must exist on the loaded robot.

The stock ``g1.xml`` welds ``waist_roll_link`` / ``torso_link`` to the parent (no
``waist_roll_joint`` / ``waist_pitch_joint``), so only **27** hinge DOFs exist.
``29`` / ``27`` layout flags both resolve to this same list (CLI backward compat).
"""

from __future__ import annotations

from typing import Literal

G1_REPLAY_JOINT_NAMES_27: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

G1_REPLAY_JOINT_NAMES_29: tuple[str, ...] = G1_REPLAY_JOINT_NAMES_27


def get_g1_replay_joint_names(_layout: Literal["29", "27"]) -> tuple[str, ...]:
  """``_layout`` kept for call-site / CLI compatibility; both map to the same 27 joints."""
  return G1_REPLAY_JOINT_NAMES_27
