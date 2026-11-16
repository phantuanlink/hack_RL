import hack_rl.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg("Hack-Throw-Frisbee-Panther"); cfg.scene.num_envs = 4
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
for k, s in env.scene.sensors.items(): print(k, type(s).__name__, type(s.data).__name__)
