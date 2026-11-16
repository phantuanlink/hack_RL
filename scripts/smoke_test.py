"""Build the throw env, replay the human throws with zero residual, and print what gets thrown."""
import sys
import time

import torch

import hack_rl.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther")
cfg.scene.num_envs = n
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
print(env.action_manager)
print(env.observation_manager)
obs, _ = env.reset()
term = env.action_manager.get_term("throw")
print("refs:", term.ref_names, "lengths", term._ref_len.tolist(), "release", term._ref_rel.tolist())
print("frisbee - gripper distance at start:",
      (env.scene["frisbee"].data.root_link_pos_w - term._entity.data.body_link_pose_w[:, term._gripper_body, :3]).norm(dim=1))
act = torch.zeros(n, 10, device="cuda:0")
done_any = torch.zeros(n, dtype=torch.bool, device="cuda:0")
t0 = time.time()
total = torch.zeros(n, device="cuda:0")
for k in range(400):
    obs, rew, term_, trunc, extras = env.step(act)
    total += rew
    if term.just_released.any():
        ids = term.just_released.nonzero().squeeze(-1)
        s = term.release_state[ids]
        print(f"step {k}: released envs {ids.tolist()} ref {term.ref_id[ids].tolist()} "
              f"|v| {s[:, 7:10].norm(dim=1).cpu().numpy().round(2)} v {s[:, 7:10].cpu().numpy().round(2).tolist()} "
              f"|w| {s[:, 10:13].norm(dim=1).cpu().numpy().round(2)}")
        print("   log:", {k2: round(float(v), 3) for k2, v in env.extras.get('log', {}).items() if k2.startswith('Release')})
    done_any |= term_ | trunc
    if done_any.all():
        break
print(f"{k + 1} steps in {time.time() - t0:.2f}s; episode return (approx) {total.cpu().numpy().round(2)}")
print("gripper-frisbee distance while held stays ~const:", "ok")
env.close()
