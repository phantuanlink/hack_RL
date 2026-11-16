import torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther-Stroke"); cfg.scene.num_envs = 512; cfg.curriculum = {}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset()
t = env.action_manager.get_term("throw")
print("refs", t.ref_names, "lengths", t._ref_len.tolist(), "release", t._ref_rel.tolist(), "speed_range", t.cfg.speed_range, "shift", t.cfg.release_shift_s)
for std in (0.0, 0.3):
    env.reset(); got = torch.zeros(512, dtype=torch.bool, device="cuda:0"); sl = got.clone(); vf = []
    for _ in range(400):
        env.step(std * torch.randn(512, t.action_dim, device="cuda:0"))
        new = (t.just_released | t.just_slipped) & ~got; sl |= t.just_slipped & ~got
        if (t.just_released & ~got).any(): vf.append(t.release_state[t.just_released & ~got, 7:10])
        got |= new
        if got.all(): break
    v = torch.cat(vf) if vf else torch.zeros(1, 3, device="cuda:0")
    print(f"noise {std}: slip {sl[got].float().mean().item()*100:.1f}%, release speed (not slipped) mean {v.norm(dim=1).mean().item():.1f} m/s, forward {v[:, 0].mean().item():.1f}")
print("ok")
