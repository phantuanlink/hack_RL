import torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
torch.manual_seed(0)
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther", play=True); cfg.scene.num_envs = 8
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset()
term = env.action_manager.get_term("throw"); dt = env.step_dt
q = [term.q_target.clone()]
for _ in range(60):
    env.step(torch.sign(torch.randn(8, term.action_dim, device="cuda:0"))); q.append(term.q_target.clone())
Q = torch.stack(q); v = (Q[1:] - Q[:-1]) / dt; acc = (v[1:] - v[:-1]) / dt
jump = (Q[1:] - Q[:-1]).abs().amax(-1) > term._vmax.max() * dt * 1.01
j2 = jump[1:] | jump[:-1]
r = (acc.abs() / term._amax)
r[j2] = 0
i = torch.nonzero(r > 1.001)
print("vmax", term._vmax.tolist(), "amax", term._amax.tolist())
print("violations (t, env, joint):", i[:10].tolist())
for t_, e_, j_ in i[:3].tolist():
    print("q", Q[t_:t_+3, e_, j_].tolist(), "lim", term._qmin[j_].item(), term._qmax[j_].item(), "ratio", r[t_, e_, j_].item())
