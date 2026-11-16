import torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther-Aggro"); cfg.scene.num_envs = 256
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset()
t = env.action_manager.get_term("throw")
obs = env.observation_manager.compute()
vt, wt = t.targets()
print("obs", {k: tuple(v.shape) for k, v in obs.items()}, "squeeze N", round(t.grip_nominal["squeeze_force_n"], 1), "aggr", round(t.aggression.min().item(), 2), round(t.aggression.max().item(), 2), "v target range", round(vt.min().item(), 1), round(vt.max().item(), 1))
got = torch.zeros(256, dtype=torch.bool, device="cuda:0"); sl = got.clone()
for _ in range(400):
    env.step(0.3 * torch.randn(256, t.action_dim, device="cuda:0"))
    sl |= t.just_slipped & ~got; got |= t.just_released | t.just_slipped
print("slip with noise 0.3:", round(sl[got].float().mean().item() * 100, 1), "%; logged", {k: round(float(v), 3) for k, v in env.extras.get("log", {}).items() if "Release/quality" in k or "speed" in k})
print("ok")
