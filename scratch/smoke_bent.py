import torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther-IK-Bent"); cfg.scene.num_envs = 256
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset(); t = env.action_manager.get_term("throw")
mx = -9.0
for _ in range(300):
    a = torch.zeros(256, t.action_dim, device="cuda:0"); a[:, 4] = 1.0                 # push the elbow straight
    env.step(a); mx = max(mx, t.q_target[:, 4].max().item())
print("ra4 qmax", round(t._qmax[4].item(), 3), "max commanded ra4 when pushing straight:", round(mx, 3))
