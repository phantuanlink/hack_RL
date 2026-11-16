"""Roll out a trained throw policy deterministically and export what the real robot should run.

    uv run python -m hack_rl.export_rollout                         # latest checkpoint of the latest run
    uv run python -m hack_rl.export_rollout --checkpoint logs/rsl_rl/panther_frisbee_throw/<run>/model_1499.pt
    uv run python -m hack_rl.export_rollout --zero                  # baseline: the adapted human throw

Writes, per reference throw, into --out (default exports/<run>/):
  throw_<ref>.csv    t, joint_ra0..joint_ra7 at 50 Hz: the joint targets the policy commanded, followed by
                     a deceleration to rest. The same format as HEADBands' joints_*.csv, for
                       python3 cartesian_follow.py --arm right --joints throw_<ref>.csv --dry-run
  throw_<ref>.json   gripper-open time and joint configuration (where to send the open command),
                     release state and its score, peak joint speed
  throw_<ref>.png    joint targets vs the reference, with the release marked
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import hack_rl.tasks  # noqa: F401
from hack_rl.robot.profile import get_profile
from hack_rl.tasks.throw import ROBOT_OF_TASK, TASK_ID
from hack_rl.tasks.throw.mdp import throw_quality
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply

ROOT = Path(__file__).resolve().parents[2]


def latest_checkpoint(log_root=ROOT / "logs" / "rsl_rl" / "panther_frisbee_throw", task: str | None = None):
    if task is not None:
        log_root = ROOT / "logs" / "rsl_rl" / get_profile(ROBOT_OF_TASK[task]).experiment
    runs = sorted(p for p in Path(log_root).glob("*") if p.is_dir() and list(p.glob("model_*.pt")))
    if not runs:
        raise FileNotFoundError(f"no checkpoints under {log_root}")
    ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1]


def spline_peak_speed(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-joint peak speed of the monotone cubic (Fritsch-Butland) spline that HEADBands'
    cartesian_follow.py sends to the joint trajectory controller (same formula as its timed()/spline_peak())."""
    d = np.diff(q, axis=0) / np.diff(t)[:, None]
    same = d[:-1] * d[1:] > 0
    vels = np.zeros_like(q)
    vels[1:-1] = np.where(same, 2 * d[:-1] * d[1:] / (d[:-1] + d[1:] + ~same), 0.0)
    s = np.linspace(0.0, 1.0, 21)[:, None, None]
    h = np.diff(t)[None, :, None]
    v = (6 * s**2 - 6 * s) * (q[:-1] - q[1:]) / h + (3 * s**2 - 4 * s + 1) * vels[:-1] + (3 * s**2 - 2 * s) * vels[1:]
    return np.abs(v).max(axis=(0, 1))


def decelerate(q: np.ndarray, dt: float, vmax: np.ndarray, amax: float, lo, hi) -> np.ndarray:
    """Append samples that bring every joint from its last velocity to rest at <= amax."""
    v = np.clip((q[-1] - q[-2]) / dt, -vmax, vmax)
    out = [q[-1].copy()]
    for _ in range(int(5.0 / dt)):
        if np.all(np.abs(v) < 1e-3):
            break
        dv = np.clip(-v, -amax * dt, amax * dt)
        v = v + dv
        nxt = np.clip(out[-1] + v * dt, lo, hi)
        v = np.where((nxt <= lo) | (nxt >= hi), 0.0, v)
        out.append(nxt)
    out += [out[-1]] * 10                       # hold 0.2 s at rest
    return np.concatenate([q, np.array(out[1:])])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=TASK_ID, help="Hack-Throw-Frisbee-Panther or Hack-Throw-Frisbee-Franka")
    ap.add_argument("--checkpoint", help="model_*.pt (default: latest of the task)")
    ap.add_argument("--pitch", type=float, help="deg, nose up (pitch-conditioned tasks)")
    ap.add_argument("--aggression", type=float, help="0..1: how hard to throw (aggression-conditioned tasks)")
    ap.add_argument("--heading", type=float, help="target heading in deg (+ = left); heading-conditioned tasks")
    ap.add_argument("--zero", action="store_true", help="no policy: replay the adapted human throw")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--decel", type=float, default=8.0, help="rad/s^2 for the stop after the throw")
    ap.add_argument("--speed-margin", type=float, default=0.99,
                    help="stretch time until the controller spline peaks at <= this fraction of the limit")
    a = ap.parse_args(argv)

    env_cfg = load_env_cfg(a.task, play=True)
    agent_cfg = load_rl_cfg(a.task)
    prof = get_profile(ROBOT_OF_TASK[a.task])
    term_cfg = env_cfg.actions["throw"]
    if a.heading is not None:
        term_cfg.heading_fixed = a.heading
    if a.aggression is not None:
        term_cfg.aggression_fixed = a.aggression
    if a.pitch is not None:
        term_cfg.pitch_fixed = a.pitch
    from hack_rl.reference import load_references
    refs, *_ = load_references(term_cfg.reference_files or prof.reference_files)
    R = len(refs)
    env_cfg.scene.num_envs = R
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if a.zero:
        ckpt = None
        n_act = env.action_manager.total_action_dim
        policy = lambda obs: torch.zeros(R, n_act, device=device)  # noqa: E731
        tag = "zero_policy"
    else:
        ckpt = Path(a.checkpoint) if a.checkpoint else latest_checkpoint(task=a.task)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        tag = f"{ckpt.parent.name}_{ckpt.stem}"
    out = Path(a.out) if a.out else ROOT / "exports" / tag
    out.mkdir(parents=True, exist_ok=True)

    term = env.action_manager.get_term("throw")
    obs = wrapped.get_observations()
    dt = env.step_dt
    q0 = term.q_target.clone()
    log = {i: {"t": [0.0], "q_target": [q0[i].cpu().numpy()], "q_ref": [term.q_ref[i].cpu().numpy()]}
           for i in range(R)}
    finished = torch.zeros(R, dtype=torch.bool, device=device)
    info = {i: {} for i in range(R)}
    steps = int(env_cfg.episode_length_s / dt) + 2
    with torch.no_grad():
        for k in range(steps):
            act = policy(obs)
            # record what is commanded in this step before stepping (the step may auto-reset finished envs)
            wrapped_step = wrapped.step(act)
            obs, _, dones, _ = wrapped_step
            for i in range(R):
                if finished[i]:
                    continue
                if dones[i]:
                    # this step ended the episode and auto-reset the env: its state is the next episode's
                    finished[i] = True
                    continue
                log[i]["t"].append((k + 1) * dt)
                log[i]["q_target"].append(term.q_target[i].cpu().numpy())
                log[i]["q_ref"].append(term.q_ref[i].cpu().numpy())
                if term.just_released[i] or (term.release_time[i] < float("inf") and "release" not in info[i]):
                    s = term.release_state[i:i + 1]
                    n = quat_apply(s[:, 3:7], torch.tensor([[0.0, 0.0, 1.0]], device=device))
                    q, comps = throw_quality(s[:, 7:10], s[:, 10:13], n, term.cfg.goal, term.forward[i:i + 1], *[x if x is None else x[i:i + 1] for x in term.targets()])
                    info[i]["release"] = {
                        "gripper_open_command_t": float(term.open_cmd_time[i]),
                        "frisbee_leaves_t": float(term.release_time[i]),
                        "velocity_mps": s[0, 7:10].cpu().numpy().round(3).tolist(),
                        "angular_velocity_radps": s[0, 10:13].cpu().numpy().round(3).tolist(),
                        "disc_normal": n[0].cpu().numpy().round(3).tolist(),
                        "quality": float(q[0]),
                        "slipped_out_of_grip": bool(term.slipped[i]),
                        "pitch_achieved_deg": float(torch.rad2deg(torch.asin((-(n[0] * term.forward[i]).sum()).clamp(-1, 1)))),
                        "grip_load_max_over_capacity": float(term.grip_u_max[i]),
                        "components": {kk: float(vv[0]) for kk, vv in comps.items()},
                    }
            if finished.all():
                break

    vmax = term._vmax.cpu().numpy()                     # the profile's limits, capped by the task variant
    arm_arg = "--arm right " if prof.name == "panther" else ""
    lo = term._qmin.cpu().numpy()
    hi = term._qmax.cpu().numpy()
    summary = []
    for i in range(R):
        name = term.ref_names[i]
        t = np.array(log[i]["t"])
        q = np.array(log[i]["q_target"])
        qr = np.array(log[i]["q_ref"])
        q_full = decelerate(q, dt, vmax, a.decel, lo, hi)
        t_full = np.arange(len(q_full)) * dt
        rel = info[i].get("release", {})
        t_open = rel.get("gripper_open_command_t", float("nan"))
        k_open = int(round(t_open / dt)) if np.isfinite(t_open) else -1
        # the controller's spline can overshoot the sampled speed by a few %: stretch time just enough that
        # cartesian_follow.py's check passes (the throw plays a little slower; release speed scales by 1/stretch)
        spline = spline_peak_speed(q_full, t_full)
        stretch = max(1.0, float((spline / (a.speed_margin * vmax)).max()))
        t_full = t_full * stretch
        t_open = t_open * stretch
        spline = spline_peak_speed(q_full, t_full)
        spline_ok = bool((spline <= vmax * (1 + 1e-6)).all())
        peak = float(np.abs(np.diff(q_full, axis=0) / np.diff(t_full)[:, None]).max())
        hdr = (f"t,{prof.joint_header} (s, rad) for cartesian_follow.py {arm_arg}--joints FILE --max-joint-vel {vmax.max():.2f}\n"
               f"time stretched x{stretch:.3f} vs the simulated policy so the controller spline stays under the limit\n"
               f"RL throw policy {tag} on reference {name}; open the right gripper at t={t_open:.3f}s "
               f"(row {k_open}, joints {np.round(q_full[k_open], 4).tolist() if k_open >= 0 else 'n/a'})\n"
               f"then decelerate at {a.decel} rad/s^2 to rest; peak commanded joint speed {peak:.2f} rad/s")
        csv = out / f"throw_{name}.csv"
        np.savetxt(csv, np.c_[t_full, q_full], delimiter=",", fmt="%.6f", header=hdr, comments="# ")
        meta = {"reference": name, "policy": tag, "checkpoint": str(ckpt) if ckpt else None,
                "rate_hz": 1 / (dt * stretch), "time_stretch": stretch,
                "note": "release_* values are from the simulation at stretch 1; velocities scale ~1/time_stretch",
                "target_heading_deg": float(term_cfg.heading_fixed),
                "aggression": float(term_cfg.aggression_fixed),
                "pitch_deg": term_cfg.pitch_fixed,
                "turn_base_by_deg": -float(term_cfg.heading_fixed),
                "gripper_open_index": k_open, "gripper_open_joints": q_full[k_open].round(5).tolist() if k_open >= 0 else None,
                "peak_joint_speed_rad_s": peak, "spline_peak_speed_rad_s": spline.round(3).tolist(),
                "passes_cartesian_follow_speed_check": spline_ok, "velocity_limit_rad_s": vmax.tolist(), **rel}
        (out / f"throw_{name}.json").write_text(json.dumps(meta, indent=1))
        summary.append(meta)
        _plot(out / f"throw_{name}.png", t * stretch, q, qr, t_full, q_full, t_open, name, rel, prof.arm_joints)
        print(f"{name}: release quality {rel.get('quality', float('nan')):.3f}  v {rel.get('velocity_mps')}  "
              f"spin {rel.get('components', {}).get('spin_radps', float('nan')):.2f} rad/s  "
              f"flat/attitude {rel.get('components', {}).get('flat', float('nan')):.2f}  "
              f"pitch {rel.get('pitch_achieved_deg', float('nan')):+.1f} deg (cmd {term_cfg.pitch_fixed})  open gripper at t={t_open:.2f}s  "
              f"time x{stretch:.3f}  "
              f"peak joint speed {peak:.2f} rad/s, controller spline peak {spline.max():.2f} rad/s "
              f"({'OK' if spline_ok else 'EXCEEDS ' + str(vmax.max()) + ': cartesian_follow will refuse'}) -> {csv}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    env.close()


def _plot(path, t, q, qr, t_full, q_full, t_open, name, rel, joints):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for j in range(len(joints)):
        a = ax[j // 4, j % 4]
        a.plot(t, qr[:, j], "--", c="gray", label="reference (adapted human)")
        a.plot(t_full, q_full[:, j], c="tab:blue", label="policy target + stop")
        if np.isfinite(t_open):
            a.axvline(t_open, c="tab:orange", lw=1, label="gripper open")
        a.set_title(joints[j], fontsize=9)
    ax[0, 0].legend(fontsize=7)
    c = rel.get("components", {})
    fig.suptitle(f"{name}: quality {rel.get('quality', float('nan')):.2f}, v_fwd {c.get('v_forward_mps', float('nan')):.2f} m/s, "
                 f"spin {c.get('spin_radps', float('nan')):.1f} rad/s, flat {c.get('flat', float('nan')):.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
