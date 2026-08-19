PYTHONPATH=src /cpfs/user/huangtao/ruofei/project/mjlab-tennis/.venv/bin/python -m mjlab.scripts.train  Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/dataset/player1/p1_serve.npz \
  --agent.run_name stage1_p1_open \
  --agent.save-interval 2000 \
  --agent.max-iterations 28000 \
  --agent.upload-model False


PYTHONPATH=src /cpfs/user/huangtao/ruofei/project/mjlab-tennis/.venv/bin/python -m mjlab.scripts.play  Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --checkpoint-file /cpfs/user/huangtao/ruofei/project/mjlab-tennis/logs/rsl_rl/g1_serve_tracking_stage1_random_dt/2026-07-24_16-27-52_stage1_p3_extreme/model_16000.pt \
  --motion-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/dataset/player3/p3_serve.npz \
  --racket-hand right


PYTHONPATH=src /cpfs/user/huangtao/ruofei/project/mjlab-tennis/.venv/bin/python -m mjlab.scripts.play Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --checkpoint-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/logs/rsl_rl/g1_serve_tracking_stage1_random_dt/2026-08-19_20-54-04_stage1_p1_open/model_8000.pt \
  --motion-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/dataset/player1/p1_serve.npz \
  --racket-hand left

uv run play Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --checkpoint-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/logs/rsl_rl/g1_serve_tracking_stage1_random_dt/stage1_p2_dt/model_27999.pt \
  --motion-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/dataset/player2/p2_serve.npz \
  --racket-hand right
  
uv run play Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --checkpoint-file /cpfs/user/huangtao/ruofei/project/mjlab-tennis/logs/rsl_rl/g1_serve_tracking_stage1_random_dt/2026-07-29_02-02-56_stage1_deyue_729/model_19999.pt \
  --motion-file /cpfs/user/huangtao/ruofei/project/AdaPT_Tennis/dataset/player3/p3_serve.npz \
  --racket-hand right

  PYTHONPATH=src /cpfs/user/huangtao/ruofei/project/mjlab-tennis/.venv/bin/python -m mjlab.scripts.play 