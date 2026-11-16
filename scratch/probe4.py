import torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(TASK_IDS["franka"], play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
V = []
orig = t._check_slip
def spy():
    orig(); V.append(torch.cat([env.sim.data.qvel[1, :9], env.sim.data.qvel[1, 9:15]]).cpu())
t._check_slip = spy
for k in range(20):
    env.step(torch.zeros(2, t.action_dim, device="cuda:0"))
V = torch.stack(V)[4:]
d = V[1:] - V[:-1]
print("qvel rms", V.pow(2).mean(0).sqrt().numpy().round(3))
print("step change rms", d.pow(2).mean(0).sqrt().numpy().round(3))
print("nv", env.sim.mj_model.nv, [env.sim.mj_model.joint(i).name for i in range(env.sim.mj_model.njnt)])
print("dof armature", env.sim.mj_model.dof_armature.round(3), "damping", env.sim.mj_model.dof_damping.round(2))
print("act gains", env.sim.mj_model.actuator_gainprm[:, 0].round(1), env.sim.mj_model.actuator_biasprm[:, :3].round(1))
