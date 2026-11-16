import sys, torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
robot = sys.argv[1]; play = sys.argv[2] == "play"
cfg = load_env_cfg(TASK_IDS[robot], play=play); cfg.scene.num_envs = 16
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
env.reset()
print("after reset: mu", t.mu[:4].tolist(), "eq std", float(env.sim.model.eq_data[:, t._weld, 3:6].std(0).sum()) if env.sim.model.eq_data.shape[0] > 1 else "shared")
for k in range(300):
    env.step(torch.zeros(16, t.action_dim, device="cuda:0"))
    if k % 25 == 0 or t.just_slipped.any():
        q = t._entity.data.joint_pos[:, t._arm_ids]; err = (q - t.q_target).abs().max(1).values
        F, tau = t.grip_wrench()
        print(f"k={k:3d} phase {t.phase[0]:.2f} track_err {err[0]:.3f} vel {t._entity.data.joint_vel[0, t._arm_ids].abs().max():.2f} u {t.grip_u[0]:.2f} |F| {F[0].norm():.1f} slipped {int(t.slipped.sum())}")
    if t.slipped.all(): break
