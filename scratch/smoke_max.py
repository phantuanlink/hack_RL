import torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther-Max"); cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset()
t = env.action_manager.get_term("throw")
print("goal", t.cfg.goal.speed, t.cfg.goal.spin, t.cfg.goal.weights, "rewards", list(env.reward_manager._term_names), "vmax", t._vmax.tolist())
for _ in range(200):
    env.step(torch.sign(torch.randn(64, t.action_dim, device="cuda:0")))
print("ok")
