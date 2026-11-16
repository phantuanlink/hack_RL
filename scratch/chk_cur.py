import hack_rl.tasks
from mjlab.tasks.registry import load_env_cfg
c = load_env_cfg("Hack-Throw-Frisbee-Panther-Max")
print(c.curriculum["grip_capacity"].params["stages"])
