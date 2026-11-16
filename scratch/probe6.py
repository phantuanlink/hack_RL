import torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(TASK_IDS["franka"], play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
rows = []
orig = t._check_slip
def spy():
    orig(); F, tau = t.grip_wrench(); d = env.sim.data
    rows.append([float(F[1, 2]), *d.qvel[1, :9].tolist(), float(d.nacon[0]) if hasattr(d, "nacon") else -1, *d.qacc[1, 9:15].tolist()])
t._check_slip = spy
for k in range(6):
    env.step(torch.zeros(2, t.action_dim, device="cuda:0"))
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print("Fz | qvel j1..7 f1 f2 | nacon | frisbee qacc lin(3) ang(3)")
print(np.array(rows)[:8])
import mujoco as mj
print("weld data", env.sim.model.eq_data[0, t._weld].tolist(), "solref", m.eq_solref[t._weld], "solimp", m.eq_solimp[t._weld])
print("frisbee body mass", m.body_mass[m.body("frisbee/frisbee").id], "inertia", m.body_inertia[m.body("frisbee/frisbee").id])
print("timestep", m.opt.timestep, "hand mass", m.body_mass[m.body("robot/hand").id])
m = env.sim.mj_model
print("geoms colliding:", [m.geom(i).name or f"g{i}:{m.body(m.geom_bodyid[i]).name}" for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]])
