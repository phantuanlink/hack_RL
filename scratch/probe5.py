import torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(TASK_IDS["franka"], play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
A = []
orig = t._check_slip
def spy():
    orig(); A.append(env.sim.data.actuator_force[1, :8].cpu())
t._check_slip = spy
for k in range(20):
    env.step(torch.zeros(2, t.action_dim, device="cuda:0"))
A = torch.stack(A)
print("actuator force abs max", A.abs().max(0).values.numpy().round(1))
print("first 12 substeps j3,j5,finger:\n", A[:12, [2, 4, 7]].numpy().round(1))
print("ref q0 vs ref q(0.2s):", t._ref_q[1, 0].cpu().numpy().round(2), t._ref_q[1, 20].cpu().numpy().round(2))
