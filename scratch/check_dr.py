import torch
import hack_rl.tasks  # noqa
from hack_rl.tasks.throw import TASK_IDS
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
cfg = load_env_cfg(TASK_IDS["panther"]); cfg.scene.num_envs = 256
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
env.reset()
t = env.action_manager.get_term("throw"); m = env.sim.model
def rng(x): return f"{float(x.min()):.4g}..{float(x.max()):.4g}"
print("mu", rng(t.mu), "| squeeze N", rng(t.squeeze), "| pad_t", rng(t.pad_t))
print("weld relpos spread (std per axis)", m.eq_data[:, t._weld, 3:6].std(0).tolist(), "matches term:", torch.allclose(m.eq_data[:, t._weld, 3:6], t.rel_pos))
print("pinch radius", rng(t.pinch_d[:, :2].norm(dim=1)), "| tilt deg", rng(torch.rad2deg(2 * torch.acos((t.rel_quat * t._rel_quat0).sum(1).abs().clamp(max=1)))))
print("frisbee mass", rng(m.body_mass[:, int(t._frisbee.indexing.body_ids[0])]))
print("joint offset", rng(t.joint_offset), "| arm frictionloss", rng(m.dof_frictionloss[:, t._entity.indexing.joint_v_adr[t._arm_ids]]))
print("kp pos elem", rng(m.actuator_gainprm[:, 0, 0]), "| kd vel elem", rng(m.actuator_gainprm[:, 8, 0]) if m.actuator_gainprm.shape[1] > 8 else "")
acts = t._entity.actuators
print("command delay (physics steps):", [(type(a).__name__, getattr(a.cfg, "delay_max_lag", None)) for a in acts[:2]])
