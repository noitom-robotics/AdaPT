"""Temporary utility: rescale generated_hit_times.pkl by a fixed multiplier.

Default use-case:
- Existing generated hit frames were exported without 30fps->50fps conversion.
- Fix by multiplying all hit-frame fields by 5/3.

This script updates in place and writes a backup alongside each pickle.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def _scale_frame(val: Any, scale: float) -> Any:
  if isinstance(val, bool):
    return val
  if isinstance(val, int):
    return int(round(float(val) * scale))
  if isinstance(val, float):
    return int(round(val * scale))
  return val


def _scale_clip_record(rec: dict[str, Any], scale: float) -> None:
  for key in ("hit_frame", "raw_hit_frame", "rollout_hit_frame", "hit_frame_relative"):
    if key in rec:
      rec[key] = _scale_frame(rec[key], scale)


def _scale_payload(payload: Any, scale: float) -> tuple[Any, int]:
  """Return (possibly modified payload, number_of_scaled_fields)."""
  changed = 0
  if isinstance(payload, dict):
    # v2 format: {"hits": {...}, "clips": [...]}
    if "hits" in payload and isinstance(payload["hits"], dict):
      for k, v in list(payload["hits"].items()):
        nv = _scale_frame(v, scale)
        if nv != v:
          changed += 1
        payload["hits"][k] = nv
    if "clips" in payload and isinstance(payload["clips"], list):
      for rec in payload["clips"]:
        if isinstance(rec, dict):
          before = rec.copy()
          _scale_clip_record(rec, scale)
          for k in ("hit_frame", "raw_hit_frame", "rollout_hit_frame", "hit_frame_relative"):
            if rec.get(k) != before.get(k):
              changed += 1
    # flat mapping fallback
    if "hits" not in payload:
      for k, v in list(payload.items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
          nv = _scale_frame(v, scale)
          if nv != v:
            changed += 1
          payload[k] = nv
  return payload, changed


def _process_one(pkl_path: Path, scale: float, dry_run: bool) -> None:
  with pkl_path.open("rb") as f:
    payload = pickle.load(f)

  payload, changed = _scale_payload(payload, scale)
  print(f"[fix] {pkl_path} fields_changed={changed}")

  if dry_run or changed == 0:
    return

  backup = pkl_path.with_suffix(pkl_path.suffix + ".bak_before_5o3")
  if not backup.exists():
    backup.write_bytes(pkl_path.read_bytes())
    print(f"[fix] backup -> {backup}")
  else:
    print(f"[fix] backup exists, skip backup write: {backup}")

  tmp = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
  with tmp.open("wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
  tmp.replace(pkl_path)
  print(f"[fix] updated -> {pkl_path}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--targets",
    nargs="+",
    default=[
      "/cpfs/user/huangtao/mjlab-tennis/dataset/0326/motion_corrected_0.75/generated_hit_times.pkl",
      "/cpfs/user/huangtao/mjlab-tennis/dataset/0326/motion_corrected/generated_hit_times.pkl",
    ],
    help="One or more generated_hit_times.pkl paths.",
  )
  parser.add_argument("--scale", type=float, default=5.0 / 3.0)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()

  for t in args.targets:
    p = Path(t).expanduser().resolve()
    if not p.is_file():
      print(f"[skip] missing: {p}")
      continue
    _process_one(p, args.scale, args.dry_run)


if __name__ == "__main__":
  main()

