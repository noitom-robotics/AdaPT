"""Generate pkl_motion_labels.json for a motion directory from an existing labels json.

This is useful when clip prefixes differ (e.g. clip_001_...) but the underlying
motion stems match after normalizing away the prefix ``clip_XXX_`` and stripping
``_rollout`` from exported npz names.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import tyro


def _normalize_clip_stem(stem: str) -> str:
  m = re.match(r"^clip_\d+_(.+)$", stem)
  return m.group(1) if m else stem


@dataclass
class GenerateLabelsCfg:
  source_labels_json: Path
  motion_dir: Path
  output_json: Path | None = None
  motion_glob: str = "*.npz"


def run(cfg: GenerateLabelsCfg) -> Path:
  src = cfg.source_labels_json.expanduser().resolve()
  if not src.is_file():
    raise FileNotFoundError(f"source_labels_json not found: {src}")
  motion_dir = cfg.motion_dir.expanduser().resolve()
  if not motion_dir.is_dir():
    raise NotADirectoryError(f"motion_dir is not a directory: {motion_dir}")

  raw = json.loads(src.read_text(encoding="utf-8"))
  if not isinstance(raw, dict):
    raise TypeError(f"Expected dict JSON at {src}, got {type(raw).__name__}")

  norm_to_label: dict[str, object] = {}
  for k, v in raw.items():
    stem = Path(str(k)).stem
    norm_to_label[_normalize_clip_stem(stem)] = v

  out: dict[str, object] = {}
  missing: list[str] = []
  for p in sorted(motion_dir.glob(cfg.motion_glob)):
    if not p.is_file():
      continue
    stem = p.stem
    if stem.endswith("_rollout"):
      stem = stem.rsplit("_rollout", 1)[0]
    n = _normalize_clip_stem(stem)
    lab = norm_to_label.get(n)
    if lab is None:
      missing.append(p.name)
      continue
    out[f"{stem}.pkl"] = lab

  if missing:
    raise KeyError(
      f"Missing labels for {len(missing)} motion file(s) under {motion_dir}. "
      f"First few: {missing[:10]}"
    )

  out_path = (
    cfg.output_json.expanduser().resolve()
    if cfg.output_json is not None
    else motion_dir / "pkl_motion_labels.json"
  )
  out_path.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n", encoding="utf-8")
  print(f"[OK] wrote {out_path} ({len(out)} entries)")
  return out_path


def main() -> None:
  cfg = tyro.cli(GenerateLabelsCfg)
  run(cfg)


if __name__ == "__main__":
  main()

