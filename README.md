# AdaPT Tennis

Stage-1 **adaptive motion-dt** tracking for tennis-serve imitation on [mjlab](https://github.com/mujocolab/mjlab).

This repository open-sources the **first-stage tracking** stack from our G1 tennis-serve work:

- Motion mimic rewards / terminations (root, body, wrist joints, hit-arm keyframes)
- Random motion-``dt`` curriculum: the environment samples ``delta_t`` each step; the policy stays **joint-only**
- Reward scaling by motion ``dt`` so clip-time totals stay comparable
- Unitree G1 with builtin position actuators (left/right racket MJCF, **no gripper / no dexhand**)

It does **not** include stage-2 serve training (ball toss/hit, residual ``dt`` policy, gripper, or dexhand).

## Tasks

| Task ID | Description |
| --- | --- |
| `Mjlab-AdaPT-Tennis-Flat-Unitree-G1` | Stage-1 random-dt tracking (default) |
| `Mjlab-AdaPT-Tennis-Flat-Unitree-G1-No-State-Estimation` | Same, without privileged actor terms |
| `Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt` | Backward-compatible alias of the default task |

## Install

```bash
git clone https://github.com/TaoHuang13/AdaPT_Tennis.git && cd AdaPT_Tennis
uv sync
```

虚拟环境默认装在项目目录下的 `.venv`。Linux 默认解析 **torch 2.9.x + cu128**（与 CUDA 12.x 驱动匹配），具体版本见仓库里的 `uv.lock`。

If `uv sync` fails with `No module named 'distutils'` (common with conda and Python 3.12+):

```bash
unset SETUPTOOLS_USE_DISTUTILS
uv sync
```

mjlab requires an NVIDIA GPU for training. We have demonstrated the training and inference pipeline in nvidia 4090 GPU.

## Train / play

Provide a motion clip (``.npz`` with ``joint_pos``, ``body_pos_w``, …):

```bash
uv run train Mjlab-AdaPT-Tennis-Flat-Unitree-G1 \
  --env.commands.motion.motion-file /path/to/serve.npz \
  --env.scene.num-envs 4096 \
  --agent.run_name stage1_random_dt
```

```bash
uv run play Mjlab-AdaPT-Tennis-Flat-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --motion-file /path/to/serve.npz \
  --racket-hand left
```

`--racket-hand` is ``left`` or ``right`` (XML + hit-arm rewards). Omit it to keep the task default (left).

List registered tasks:

```bash
uv run list_envs
```

## Layout

```
src/mjlab/tasks/adapt_tennis/
  stage1_tracking_env_cfg.py   # MDP terms (obs / rewards / terminations)
  mdp/commands.py              # MotionCommand + dynamic / random dt
  config/g1/                   # G1 wiring + PPO runner
```

## License

Apache License 2.0 (see `LICENSE`). Built on mjlab; see file headers for third-party licenses.
