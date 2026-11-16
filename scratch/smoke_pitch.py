import math, torch
import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from hack_rl.tasks.throw.mdp import throw_quality, ThrowGoal
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther-Stroke-Pitch"); cfg.scene.num_envs = 128
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); env.reset(); t = env.action_manager.get_term("throw")
obs = env.observation_manager.compute()
print("obs", {k: tuple(v.shape) for k, v in obs.items()}, "pitch deg range", round(math.degrees(t.pitch.min().item()), 1), round(math.degrees(t.pitch.max().item()), 1))
vt, wt, nt = t.targets(); i = int(torch.argmax(t.pitch))
print("env with max pitch: forward", t.forward[i].tolist(), "target normal", [round(x, 3) for x in nt[i].tolist()])
for _ in range(100): env.step(torch.zeros(128, t.action_dim, device="cuda:0"))
# reward sanity: disc tilted exactly as commanded scores attitude 1, flat disc at 15 deg command scores less
g = ThrowGoal(); f = torch.tensor([[1.0, 0, 0]]); v = torch.tensor([[5.0, 0, 0.4]]); w = torch.tensor([[0, 0, 15.0]])
p = math.radians(15); n_cmd = torch.tensor([[-math.sin(p), 0, math.cos(p)]])
for name, n in (("tilted as commanded", n_cmd), ("flat", torch.tensor([[0, 0, 1.0]]))):
    q, c = throw_quality(v, w, n, g, f, None, None, n_cmd)
    print(f"15 deg command, disc {name}: attitude {c['flat'].item():.2f}, quality {q.item():.3f}")
print("ok")
