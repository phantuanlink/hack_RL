import sys, torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
robot = sys.argv[1]; play = len(sys.argv) > 2
TASK_IDS = dict(TASK_IDS, safe="Hack-Throw-Frisbee-Panther-Safe", mid="Hack-Throw-Frisbee-Panther-Mid")
cfg = load_env_cfg(TASK_IDS[robot], play=play); cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
print("action dim", t.action_dim, "obs", {k: v.shape for k, v in env.observation_manager.compute().items()})
print("eq_data per env distinct:", env.sim.model.eq_data.shape, float(env.sim.model.eq_data[:, t._weld, 3:6].std(0).sum()))
print("frisbee mass range", env.sim.model.body_mass[:, int(t._frisbee.indexing.body_ids[0])].aminmax())
print("mu", t.mu.aminmax(), "squeeze", t.squeeze.aminmax())
obs, _ = env.reset()
umax = torch.zeros(64, device="cuda:0"); rel = torch.zeros(64, dtype=torch.bool, device="cuda:0"); sl = rel.clone()
for k in range(400):
    obs, r, term, trunc, ext = env.step(torch.zeros(64, t.action_dim, device="cuda:0"))
    umax = torch.maximum(umax, t.grip_u); rel |= t.just_released; sl |= t.just_slipped
    if k == 5:
        F, tau = t.grip_wrench(); print("held wrench at rest: |F|", F.norm(dim=1)[:4].tolist(), "|tau|", tau.norm(dim=1)[:4].tolist(), "u", t.grip_u[:4].tolist())
print("cap scale", t.capacity_scale, "grip u max p50/max", umax.median().item(), umax.max().item(), "released", rel.float().mean().item(), "slipped", sl.float().mean().item())
print("rewards", {k: float(v) for k, v in ext.get("log", {}).items() if "Grip" in k or "Release/quality" in k})
