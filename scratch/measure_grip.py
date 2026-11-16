"""Grip wrench the v3 policy demands at the pinch point (from the weld's constraint force)."""
import json, math
from dataclasses import asdict
import numpy as np, torch
import hack_rl.tasks  # noqa
from hack_rl.export_rollout import latest_checkpoint
from hack_rl.tasks.throw import TASK_ID
from hack_rl.robot.panther import GRIP_REL_POS, GRIP_REL_QUAT, LINK_RF_IN_RA7
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

N = 1024
cfg = load_env_cfg(TASK_ID); cfg.scene.num_envs = N; cfg.curriculum = {}
agent = load_rl_cfg(TASK_ID); dev = "cuda:0"
env = ManagerBasedRlEnv(cfg=cfg, device=dev)
w = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
r = MjlabOnPolicyRunner(w, asdict(agent), device=dev)
r.load(str(latest_checkpoint()), load_cfg={"actor": True}, strict=True, map_location=dev)
pol = r.get_inference_policy(device=dev)
t = env.action_manager.get_term("throw")
fr = t._frisbee; vadr = fr.data.indexing.free_joint_v_adr
relq = torch.tensor(GRIP_REL_QUAT, device=dev)[None]
pg = torch.tensor(LINK_RF_IN_RA7 + np.array([0, 0, -0.065]), device=dev, dtype=torch.float32)[None]
p_d = quat_apply_inverse(relq, pg - torch.tensor(GRIP_REL_POS, device=dev)[None])[0]
print("pinch point in disc frame", p_d.tolist())
mx = {k: torch.zeros(N, device=dev) for k in ["Ft", "Fn", "tau_n", "tau_t"]}
done_any = torch.zeros(N, dtype=torch.bool, device=dev)
obs = w.get_observations()
with torch.no_grad():
    for _ in range(400):
        obs, _, d, _ = w.step(pol(obs))
        f = env.sim.data.qfrc_constraint[:, vadr]          # (N, 6): force world, torque body
        quat = fr.data.root_link_quat_w
        F = quat_apply_inverse(quat, f[:, :3]); tau = f[:, 3:6] - torch.cross(p_d.expand(N, 3), F, dim=1)
        live = t.held & ~done_any
        vals = {"Ft": F[:, :2].norm(dim=1), "Fn": F[:, 2].abs(), "tau_n": tau[:, 2].abs(), "tau_t": tau[:, :2].norm(dim=1)}
        for k in mx: mx[k] = torch.where(live, torch.maximum(mx[k], vals[k]), mx[k])
        done_any |= d.bool()
        if done_any.all(): break
res = {k: {"p50": float(v.median()), "p90": float(v.quantile(0.9)), "max": float(v.max())} for k, v in mx.items()}
print(json.dumps(res, indent=1))
