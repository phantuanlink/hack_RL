import sys, torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
robot, ts = sys.argv[1], float(sys.argv[2])
cfg = load_env_cfg(TASK_IDS[robot], play=True); cfg.scene.num_envs = 2
cfg.sim.mujoco.timestep = ts; cfg.decimation = int(round(0.02 / ts))
if len(sys.argv) > 3: cfg.sim.mujoco.iterations = int(sys.argv[3]); cfg.sim.mujoco.ls_iterations = 2 * int(sys.argv[3])
print("solver", cfg.sim.mujoco.solver if hasattr(cfg.sim.mujoco, "solver") else "?", cfg.sim.mujoco.iterations)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
Fs = []
orig = t._check_slip
def spy():
    orig(); F, tau = t.grip_wrench(); Fs.append(torch.cat([F[1], tau[1]]).cpu())
t._check_slip = spy
for k in range(40):
    env.step(torch.zeros(2, t.action_dim, device="cuda:0"))
F = torch.stack(Fs)[4:]
d = F[1:] - F[:-1]
print(robot, ts, "integrator", env.sim.mj_model.opt.integrator, "rms F", F.pow(2).mean(0).sqrt().numpy().round(2), "rms step-to-step change", d.pow(2).mean(0).sqrt().numpy().round(2), "slipped", t.slipped.tolist(), "u_max", t.grip_u_max.tolist())
