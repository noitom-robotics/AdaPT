"""Plot every dimension in result txt logs to PNG files.

Input txt format (per line):
  step env_id v0 v1 v2 ...
Header/comment lines starting with "#" are ignored.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rows: list[list[float]] = []
  with path.open("r", encoding="utf-8") as f:
    for line in f:
      s = line.strip()
      if not s or s.startswith("#"):
        continue
      parts = s.split()
      try:
        rows.append([float(x) for x in parts])
      except ValueError:
        continue
  if not rows:
    raise ValueError(f"No numeric rows found in {path}")
  # Keep rows with the dominant column count (robust to mixed log lines).
  lens = Counter(len(r) for r in rows)
  target_len, _ = lens.most_common(1)[0]
  rows = [r for r in rows if len(r) == target_len]
  arr = np.asarray(rows, dtype=np.float64)
  if arr.shape[1] < 3:
    raise ValueError(f"Expected >=3 columns (step env values...) in {path}, got {arr.shape[1]}")
  steps = arr[:, 0].astype(np.int64)
  env_ids = arr[:, 1].astype(np.int64)
  vals = arr[:, 2:]
  return steps, env_ids, vals


def _pick_env_series(
  steps: np.ndarray, env_ids: np.ndarray, vals: np.ndarray, env_id: int
) -> tuple[np.ndarray, np.ndarray]:
  m = env_ids == env_id
  if not np.any(m):
    raise ValueError(f"env_id={env_id} not found")
  s = steps[m]
  y = vals[m]
  order = np.argsort(s)
  return s[order], y[order]


def plot_one_file(path: Path, out_root: Path) -> int:
  steps, env_ids, vals = load_txt(path)
  out_dir = out_root / path.stem
  out_dir.mkdir(parents=True, exist_ok=True)

  uniq_envs = np.unique(env_ids)
  num_dims = vals.shape[1]
  for dim in range(num_dims):
    plt.figure(figsize=(10, 4))
    for eid in uniq_envs:
      m = env_ids == eid
      s = steps[m]
      y = vals[m, dim]
      order = np.argsort(s)
      plt.plot(s[order], y[order], linewidth=0.9, label=f"env{eid}")
    plt.xlabel("step")
    plt.ylabel(f"dim_{dim}")
    plt.title(f"{path.name} dim={dim}")
    if len(uniq_envs) <= 8:
      plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"dim_{dim:04d}.png", dpi=140)
    plt.close()
  return num_dims


def plot_compare_two_files(
  path_a: Path, path_b: Path, out_root: Path, env_id: int
) -> int:
  s_a, e_a, v_a = load_txt(path_a)
  s_b, e_b, v_b = load_txt(path_b)
  xs_a, ys_a = _pick_env_series(s_a, e_a, v_a, env_id)
  xs_b, ys_b = _pick_env_series(s_b, e_b, v_b, env_id)

  num_dims = min(ys_a.shape[1], ys_b.shape[1])
  out_dir = out_root / f"{path_a.stem}_vs_{path_b.stem}"
  out_dir.mkdir(parents=True, exist_ok=True)

  for dim in range(num_dims):
    plt.figure(figsize=(10, 4))
    plt.plot(xs_a, ys_a[:, dim], linewidth=0.9, label=path_a.stem)
    plt.plot(xs_b, ys_b[:, dim], linewidth=0.9, label=path_b.stem)
    plt.xlabel("step")
    plt.ylabel(f"dim_{dim}")
    plt.title(f"{path_a.name} vs {path_b.name} | env={env_id} | dim={dim}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"dim_{dim:04d}.png", dpi=140)
    plt.close()
  return num_dims


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=Path("logs/results"),
    help="Directory containing txt logs.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("logs/results/png"),
    help="Directory to save PNGs.",
  )
  parser.add_argument(
    "--files",
    type=str,
    nargs="*",
    default=None,
    help="Optional txt basenames to plot, e.g. mvae.txt tracking_motion_command.txt",
  )
  parser.add_argument(
    "--compare-two",
    action="store_true",
    help="Plot two txt files on the same figure per dimension.",
  )
  parser.add_argument(
    "--env-id",
    type=int,
    default=0,
    help="When --compare-two is set, compare this env_id only.",
  )
  args = parser.parse_args()

  in_dir: Path = args.input_dir
  out_dir: Path = args.output_dir
  out_dir.mkdir(parents=True, exist_ok=True)

  if args.files:
    txts = [in_dir / x for x in args.files]
  else:
    txts = sorted(in_dir.glob("*.txt"))

  txts = [p for p in txts if p.exists()]
  if not txts:
    raise FileNotFoundError(f"No txt files found under {in_dir}")

  if args.compare_two:
    if len(txts) != 2:
      raise ValueError("--compare-two requires exactly 2 txt files via --files")
    n = plot_compare_two_files(txts[0], txts[1], out_dir, args.env_id)
    print(
      f"[OK] compare {txts[0].name} vs {txts[1].name} "
      f"-> {out_dir / (txts[0].stem + '_vs_' + txts[1].stem)} ({n} dims)"
    )
    return

  for p in txts:
    n = plot_one_file(p, out_dir)
    print(f"[OK] {p} -> {out_dir / p.stem} ({n} dims)")


if __name__ == "__main__":
  main()

