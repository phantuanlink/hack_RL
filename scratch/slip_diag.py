import torch, sys
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
task = sys.argv[1]
cfg = load_env_cfg(task); cfg.scene.num_envs = 512; cfg.curriculum = {}
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
t = env.action_manager.get_term("throw")
for scale in (1.0, 1.5, 3.0):
    for std in (0.0, 0.1, 0.3):
        t.capacity_scale = scale
        env.reset()
        got = torch.zeros(512, dtype=torch.bool, device="cuda:0"); sl = got.clone(); phase_slip = []
        for _ in range(400):
            env.step(std * torch.randn(512, t.action_dim, device="cuda:0"))
            ns = t.just_slipped & ~got
            if ns.any(): phase_slip.append((t.phase / t.ref_length)[ns])
            sl |= ns; got |= t.just_released | t.just_slipped
            if got.all(): break
        ph = torch.cat(phase_slip).mean().item() if phase_slip else float("nan")
        print(f"{task.split('-')[-1]}: grip x{scale}, action noise {std}: slip {sl[got].float().mean().item()*100:5.1f}%  (mean slip at {ph:.2f} of the throw)")
