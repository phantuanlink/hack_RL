import sys, torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
robot = sys.argv[1]
cfg = load_env_cfg(TASK_IDS[robot], play=True); cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
q_hold = t.q_target.clone()
def frozen(actions):
    t.q_target[:] = q_hold; t.v_cmd[:] = 0; t._q_prev = q_hold.clone(); t._sub = 0
t.process_actions = frozen
rows = []
orig = t._check_slip
def spy():
    orig(); F, tau = t.grip_wrench(); rows.append(float(F[1, 2]))
t._check_slip = spy
for k in range(10):
    env.step(torch.zeros(2, t.action_dim, device="cuda:0"))
import numpy as np
print(robot, "arm frozen: Fz per substep", np.round(rows, 2).tolist())
