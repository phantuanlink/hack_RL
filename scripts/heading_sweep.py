"""Evaluate a heading-conditioned policy at fixed target headings, every randomisation on.

    uv run python scripts/heading_sweep.py --task Hack-Throw-Frisbee-Panther-Heading --checkpoint <model.pt> \
        [--headings -35 -30 ... 5] [--envs 2048]

Per heading: mean release score (a slip scores 0), slip %, speed, spin, straight-ahead, flat, peak shoulder
torque, arm push into the torso. The best heading is the one to export (`export_rollout --heading`); on
the robot, turn the base by -heading so the throw points at the target.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import hack_rl.tasks  # noqa: F401
from hack_rl.tasks.throw.mdp import ArmReaction, throw_quality
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply


def run_round(env, w, policy, term, n, dev, tau_ids, tau_lim):
    obs, _ = env.reset()
    obs = w.get_observations()
    got = torch.zeros(n, dtype=torch.bool, device=dev)
    slipped = torch.zeros(n, dtype=torch.bool, device=dev)
    rel = torch.zeros(n, 13, device=dev)
    fwd = torch.zeros(n, 3, device=dev)
    tau_peak = torch.zeros(n, device=dev)
    react = ArmReaction()
    react_peak = torch.zeros(n, device=dev)
    with torch.no_grad():
        for _ in range(int(env.cfg.episode_length_s / env.step_dt) + 2):
            obs, _, _, _ = w.step(policy(obs))
            live = ~got & (env.episode_length_buf > 5)
            tau_peak[live] = torch.maximum(tau_peak[live], (env.sim.data.qfrc_actuator[:, tau_ids].abs() / tau_lim)[live].amax(1))
            react_peak[live] = torch.maximum(react_peak[live], react(env).sqrt()[live])
            new = (term.just_released | term.just_slipped) & ~got
            rel[new] = term.release_state[new]
            fwd[new] = term.forward[new]
            slipped[new] = term.just_slipped[new]
            got |= new
            if got.all():
                break
    n_disc = quat_apply(rel[:, 3:7], torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
    q, comps = throw_quality(rel[:, 7:10], rel[:, 10:13], n_disc, term.cfg.goal, fwd, *term.targets())
    ok = got & ~slipped
    q = q * (~slipped).float()
    r = {"episodes": int(got.sum()), "quality_mean": float(q[got].mean()), "slipped_pct": float(100 * slipped[got].float().mean()),
         "quality_mean_not_slipped": float(q[ok].mean()) if ok.any() else float("nan"),
         "shoulder_elbow_peak_torque_p50": float(tau_peak[got].median()), "shoulder_elbow_peak_torque_p90": float(tau_peak[got].quantile(0.9)),
         "torque_at_limit_pct": float(100 * (tau_peak[got] > 0.98).float().mean()),
         "torso_push_p50": float(react_peak[got].median())}
    for k in ("v_forward_mps", "spin_radps", "direction", "flat"):
        r[k] = float(comps[k][ok].mean()) if ok.any() else float("nan")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="Hack-Throw-Frisbee-Panther-Heading")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--headings", nargs="+", type=float, default=[-35, -30, -25, -20, -15, -10, -5, 0, 5])
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--aggression", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_env_cfg(a.task)
    cfg.scene.num_envs = a.envs
    cfg.curriculum = {}
    cfg.actions["throw"].heading_range = None
    cfg.actions["throw"].aggression_range = None
    cfg.actions["throw"].aggression_fixed = a.aggression
    agent_cfg = load_rl_cfg(a.task)
    dev = "cuda:0"
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(w, asdict(agent_cfg), device=dev)
    runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=dev)
    policy = runner.get_inference_policy(device=dev)
    term = env.action_manager.get_term("throw")
    tau_ids = term._entity.indexing.joint_v_adr[term._arm_ids[:4]]          # shoulder + elbow (ra0-ra3)
    tau_lim = torch.tensor([x.effort_limit for x in cfg.scene.entities["robot"].articulation.actuators[:4]], device=dev)
    res = {}
    for h in a.headings:
        term.cfg.heading_fixed = h
        r = run_round(env, w, policy, term, a.envs, dev, tau_ids, tau_lim)
        res[h] = r
        print(f"heading {h:+5.0f} deg: score {r['quality_mean']:.3f} (not slipped {r['quality_mean_not_slipped']:.3f}), "
              f"slip {r['slipped_pct']:4.1f}%, v {r['v_forward_mps']:.2f} m/s, spin {r['spin_radps']:.1f}, "
              f"straight {r['direction']:.2f}, flat {r['flat']:.2f}, shoulder/elbow torque p50/p90 "
              f"{r['shoulder_elbow_peak_torque_p50']:.2f}/{r['shoulder_elbow_peak_torque_p90']:.2f} "
              f"(at limit {r['torque_at_limit_pct']:.0f}%), torso push {r['torso_push_p50']:.2f}", flush=True)
    best = max(res, key=lambda h: res[h]["quality_mean"])
    print(f"best heading {best:+.0f} deg -> export with --heading {best:.0f}; turn the base {(-best):+.0f} deg "
          f"({'left' if -best > 0 else 'right'})")
    out = Path(a.out) if a.out else Path(a.checkpoint).with_name("heading_sweep.json")
    out.write_text(json.dumps({"best_heading_deg": best, "results": res}, indent=1))
    env.close()


if __name__ == "__main__":
    main()
