import torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
for acc, res in (((30.0,) * 5 + (80.0,) * 3, (0.6, 0.6, 0.6, 0.6, 1.2, 1.5, 0.9, 1.6)), ((100.0,) * 5 + (200.0,) * 3, (0.6, 0.6, 0.6, 0.6, 1.2, 1.5, 0.9, 1.6)),
                 ((30.0,) * 5 + (80.0,) * 3, (0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.6, 1.0)),
                 ((20.0,) * 5 + (50.0,) * 3, (0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.6, 1.0))):
    cfg = load_env_cfg("Hack-Throw-Frisbee-Panther-Max"); cfg.scene.num_envs = 512; cfg.curriculum = {}
    cfg.actions["throw"].accel_cap = acc; cfg.actions["throw"].residual_scale = res
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); t = env.action_manager.get_term("throw")
    for scale in (3.0,):
        out = []
        for std in (0.0, 0.1, 0.3):
            t.capacity_scale = scale; env.reset()
            got = torch.zeros(512, dtype=torch.bool, device="cuda:0"); sl = got.clone()
            for _ in range(400):
                env.step(std * torch.randn(512, t.action_dim, device="cuda:0"))
                sl |= t.just_slipped & ~got; got |= t.just_released | t.just_slipped
                if got.all(): break
            out.append(f"noise {std}: {sl[got].float().mean().item()*100:5.1f}%")
        print(f"accel {acc[0]:.0f}/{acc[-1]:.0f}, wrist residual {res[4:]}, grip x{scale}: slip " + ", ".join(out), flush=True)
    env.close()
