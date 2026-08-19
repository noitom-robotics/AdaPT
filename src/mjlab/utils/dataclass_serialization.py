"""Helpers to log dataclass configs without deep-copying non-serializable runtime objects."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import mujoco
import numpy as np
import torch


def asdict_env_cfg_for_yaml(cfg: Any) -> dict[str, Any]:
  """Convert env config to nested dicts for YAML; safe after env build.

  Standard :func:`dataclasses.asdict` uses :func:`copy.deepcopy` on non-dataclass
  values. Tennis (and similar) attach runtime objects to the config after build
  (e.g. ``TennisLatentPipelineHandle.stack`` → :class:`~mjlab.entity.Entity`),
  which holds ``mujoco.MjSpec`` and cannot be deep-copied.
  """

  def convert(x: Any) -> Any:
    if isinstance(x, (mujoco.MjSpec, mujoco.MjModel)):
      return f"<{type(x).__name__} (not serializable)>"
    # Runtime-only handle filled in ActionTerm.build — never deep-copy stack.
    if type(x).__name__ == "TennisLatentPipelineHandle":
      return {"stack": None}
    if is_dataclass(x) and not isinstance(x, type):
      return {f.name: convert(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, dict):
      return {convert(k): convert(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
      return type(x)(convert(v) for v in x)
    if isinstance(x, (str, int, float, bool)) or x is None:
      return x
    if isinstance(x, slice):
      return str(x)
    if isinstance(x, set):
      return {convert(v) for v in x}
    if isinstance(x, frozenset):
      return frozenset(convert(v) for v in x)
    if callable(x) and not isinstance(x, type):
      mod = getattr(x, "__module__", "")
      qual = getattr(x, "__qualname__", repr(x))
      return f"{mod}.{qual}" if mod else qual
    if isinstance(x, torch.Tensor):
      return (
        x.detach().cpu().tolist()
        if x.numel() < 4096
        else f"<torch.Tensor {tuple(x.shape)} {x.dtype}>"
      )
    if isinstance(x, np.ndarray):
      return x.tolist() if x.size < 4096 else f"<ndarray {x.shape} {x.dtype}>"
    return repr(x)

  out = convert(cfg)
  assert isinstance(out, dict)
  return out
