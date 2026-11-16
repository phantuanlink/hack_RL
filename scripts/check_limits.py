"""Drive the throw env with random bang-bang actions; report the commanded targets' worst speed/acceleration."""
import torch
import hack_rl.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Hack-Throw-Frisbee-Panther", play=True)
cfg.scene.num_envs = 64
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
t = env.action_manager.get_term("throw")
dt = env.step_dt
Q = [t.q_target.clone()]
for _ in range(80):
    env.step(torch.sign(torch.randn(64, 10, device="cuda:0")))
    Q.append(t.q_target.clone())
Q = torch.stack(Q)
v = (Q[1:] - Q[:-1]) / dt
a = (v[1:] - v[:-1]) / dt
jump = ((Q[1:] - Q[:-1]).abs().amax(-1) > t._vmax.max() * dt * 1.01)
j2 = jump[1:] | jump[:-1]
ratio = (a.abs() / t._amax)
ratio[j2] = 0
k = ratio.argmax()
s, e, j = torch.unravel_index(k, ratio.shape)
print("max |v|/vmax", float((v.abs() / t._vmax)[~jump].max()), " max |a|/amax", float(ratio.max()),
      "at step", int(s), "joint", int(j), "q", float(Q[s + 1, e, j]), "limits", float(t._qmin[j]), float(t._qmax[j]))
print("fraction of (step,env,joint) over amax:", float((ratio > 1.001).float().mean()))
env.close()
