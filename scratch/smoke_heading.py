import math, torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import sys; cfg = load_env_cfg(sys.argv[1] if len(sys.argv) > 1 else "Hack-Throw-Frisbee-Panther-Heading"); cfg.scene.num_envs = 128
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset()
t = env.action_manager.get_term("throw")
obs = env.observation_manager.compute()
print("obs", {k: tuple(v.shape) for k, v in obs.items()}, "heading deg range", [round(math.degrees(x), 1) for x in (t.heading.min().item(), t.heading.max().item())])
print("lag", [(a.cfg.delay_min_lag, a.cfg.delay_max_lag) for a in t._entity.actuators[:1]], "vmax", t._vmax.tolist(), "amax", t._amax.tolist(), "delay range", t._delay_range, "latency", t.cfg.release_latency_range)
for _ in range(300):
    env.step(torch.zeros(128, t.action_dim, device="cuda:0"))
print("ok")
