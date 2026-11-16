import torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(TASK_IDS["franka"], play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
orig = t._check_slip
def spy():
    orig(); F, tau = t.grip_wrench()
    print(f"u {t.grip_u.tolist()} F1 {F[1].tolist()} tau1 {tau[1].tolist()} contacts {int(env.sim.data.nacon[0]) if hasattr(env.sim.data,'nacon') else '?'}")
t._check_slip = spy
for k in range(3):
    env.step(torch.zeros(2, t.action_dim, device="cuda:0"))
import mujoco
m = env.sim.mj_model
d = env.sim.data
print("ncon per world", d.ncon if hasattr(d, "ncon") else None)
q = t._entity.data.joint_pos[1, t._arm_ids]; print("q env1", q.tolist(), "target", t.q_target[1].tolist())
