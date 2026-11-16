"""Robustness check: run a trained policy (deterministic actions) in many randomised envs, one episode each.

    uv run python scripts/evaluate.py [--checkpoint model.pt] [--envs 2048]

Uses the *training* env config with every randomisation on (see env_cfg.py) and the real grip capacity.
The actor is open loop, so this is also how robust the exported joint CSV is. Reports the spread of the release score and its parts: how much the throw degrades if
system ID is a bit off.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import hack_rl.tasks  # noqa: F401
from hack_rl.export_rollout import latest_checkpoint
from hack_rl.tasks.throw import TASK_ID
from hack_rl.tasks.throw.mdp import throw_quality
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=TASK_ID)
    ap.add_argument("--checkpoint")
    ap.add_argument("--heading", type=float, help="fix the target heading (deg) instead of the training range")
    ap.add_argument("--aggression", type=float, help="fix the aggressiveness (0..1) instead of the training range")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--zero", action="store_true", help="evaluate the adapted human throw (no policy)")
    a = ap.parse_args()
    cfg = load_env_cfg(a.task)
    cfg.scene.num_envs = a.envs
    if a.aggression is not None:
        cfg.actions["throw"].aggression_range = None
        cfg.actions["throw"].aggression_fixed = a.aggression
    if a.heading is not None:
        cfg.actions["throw"].heading_range = None
        cfg.actions["throw"].heading_fixed = a.heading
    cfg.curriculum = {}                     # grip capacity stays at the real one (scale 1)
    agent_cfg = load_rl_cfg(a.task)
    dev = "cuda:0"
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if a.zero:
        n_act = env.action_manager.total_action_dim
        policy, tag = (lambda o: torch.zeros(a.envs, n_act, device=dev)), "zero_policy"
    else:
        ckpt = Path(a.checkpoint) if a.checkpoint else latest_checkpoint(task=a.task)
        r = MjlabOnPolicyRunner(w, asdict(agent_cfg), device=dev)
        r.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=dev)
        policy, tag = r.get_inference_policy(device=dev), f"{ckpt.parent.name}/{ckpt.stem}"
    term = env.action_manager.get_term("throw")
    obs = w.get_observations()
    got = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    rel = torch.zeros(a.envs, 13, device=dev)
    contact = torch.zeros(a.envs, device=dev)
    slipped = torch.zeros(a.envs, dtype=torch.bool, device=dev)
    u_max = torch.zeros(a.envs, device=dev)
    from hack_rl.tasks.throw.mdp import ArmReaction
    reaction = ArmReaction()
    tau_ids = term._entity.indexing.joint_v_adr[term._arm_ids]
    tau_lim = torch.tensor([a.effort_limit for a in cfg.scene.entities["robot"].articulation.actuators[:len(tau_ids)]],
                           device=dev)
    tau_peak = torch.zeros(a.envs, len(tau_ids), device=dev)       # max |torque| / limit while holding
    react_peak = torch.zeros(a.envs, device=dev)                     # max arm push into the torso, x arm weight
    sensors = [k for k, s in env.scene.sensors.items() if k.startswith("arm_contact") and hasattr(s.data, "found")]
    ref = term.ref_id.clone()
    with torch.no_grad():
        for _ in range(int(cfg.episode_length_s / env.step_dt) + 2):
            obs, _, dones, _ = w.step(policy(obs))
            new = (term.just_released | term.just_slipped) & ~got
            rel[new] = term.release_state[new]
            ref[new] = term.ref_id[new]
            slipped[new] = term.just_slipped[new]
            u_max[new] = term.grip_u_max[new]
            for sn in sensors:
                contact += (env.scene[sn].data.found.reshape(a.envs, -1).any(1) & ~got).float()
            live = ~got & (env.episode_length_buf > 5)              # skip the settling right after reset
            tau_peak[live] = torch.maximum(tau_peak[live], (env.sim.data.qfrc_actuator[:, tau_ids].abs() / tau_lim)[live])
            react_peak[live] = torch.maximum(react_peak[live], reaction(env).sqrt()[live])
            got |= new
            if got.all():
                break
    n = quat_apply(rel[:, 3:7], torch.tensor([0.0, 0.0, 1.0], device=dev).expand(a.envs, 3))
    q, comps = throw_quality(rel[:, 7:10], rel[:, 10:13], n, term.cfg.goal, term.forward, *term.targets())
    q = q * (~slipped).float()                       # a disc that slipped out was not thrown
    q, ok = q[got].cpu().numpy(), got.cpu().numpy()
    res = {"policy": tag, "episodes": int(ok.sum()), "quality_mean": float(q.mean()), "quality_p10": float(np.percentile(q, 10)),
           "quality_p50": float(np.median(q)), "quality_p90": float(np.percentile(q, 90)),
           "arm_body_contact_episodes_pct": float(100 * (contact[got] > 0).float().mean()),
           "slipped_pct": float(100 * slipped[got].float().mean()),
           "peak_torque_over_limit_per_joint_p50": tau_peak[got].median(0).values.cpu().numpy().round(3).tolist(),
           "peak_torque_over_limit_per_joint_p90": tau_peak[got].quantile(0.9, dim=0).cpu().numpy().round(3).tolist(),
           "arm_push_on_torso_x_arm_weight_p50": float(react_peak[got].median()),
           "arm_push_on_torso_x_arm_weight_p90": float(react_peak[got].quantile(0.9)),
           "grip_load_over_capacity_p50": float(u_max[got].median()), "grip_load_over_capacity_p90": float(u_max[got].quantile(0.9))}
    for k, v in comps.items():
        v = v[got & ~slipped].cpu().numpy()
        res[k] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    for r_id, name in enumerate(term.ref_names):
        m = (ref[got] == r_id).cpu().numpy()
        if m.any():
            res[f"quality_{name}"] = float(q[m].mean())
    print(json.dumps(res, indent=1))
    env.close()


if __name__ == "__main__":
    main()
