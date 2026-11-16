"""Turn the retargeted human throws into starting points for RL.

    uv run python -m hack_rl.prepare_references        # writes data/reference/adapted/*.csv + summary.json

The HEADBands retargeting matches the hand *path* of the phone videos but not a frisbee grip: at the
recorded release the disc is tilted upside down, and that release time sits in the slow-down after the
stroke. For each reference this script
  1. puts the nominal release at the moment of peak forward hand speed (the actual throwing stroke),
  2. finds a joint offset that makes the disc flat and upright at that moment while keeping the hand
     within a few cm of the human path (MuJoCo forward kinematics + scipy),
  3. blends that offset in smoothly from the start, so the throw begins exactly at the human pose,
and writes the result in the same CSV format (rows t, joint_ra0..7; a "release at t=...s" comment).
RL then fine-tunes around these (residuals, playback speed, release timing).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from hack_rl.reference import DATA_DIR, load_reference
from hack_rl.robot.panther import ROBOT_XML

SOURCE_FILES = ("joints_trajectory_shoulder_rel.csv", "joints_trajectory_shoulder_rel_2.csv",
                "joints_trajectory_shoulder_rel_3.csv")
OUT_DIR = DATA_DIR / "adapted"


class Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("grip_site").id
        self.grip = self.m.body("gripper").id
        r = self.m.jnt_range[:8]
        self.lo, self.hi = r[:, 0] * 0.95, r[:, 1] * 0.95

    def fk(self, q):
        self.d.qpos[:8] = q
        mujoco.mj_kinematics(self.m, self.d)
        normal = self.d.xmat[self.grip].reshape(3, 3)[:, 1].copy()   # disc normal = gripper y
        return self.d.site_xpos[self.site].copy(), normal


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def adapt(path: Path, kin: Kin, dt: float = 0.01, pos_tol: float = 0.05):
    ref = load_reference(path, dt)
    P = np.array([kin.fk(q)[0] for q in ref.q])
    V = np.gradient(P, dt, axis=0)
    k_rel = int(np.argmax(V[:, 0]))                       # peak forward (+x) hand speed
    q_rel = ref.q[k_rel]
    p_rel, _ = kin.fk(q_rel)

    def cost(dq):
        p, n = kin.fk(q_rel + dq)
        return (1 - n[2]) ** 2 + (np.linalg.norm(p - p_rel) / pos_tol) ** 2 * 0.2 + 0.01 * (dq @ dq)

    bounds = [(lo - q, hi - q) for lo, hi, q in zip(kin.lo, kin.hi, q_rel)]
    best = None
    rng = np.random.default_rng(0)
    for trial in range(40):                                # multi-start: the cost has several basins
        x0 = rng.uniform(-1.0, 1.0, 8) if trial else np.zeros(8)
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(cost, x0, method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    dq = best.x
    w = smoothstep(ref.t / ref.t[k_rel])[:, None]           # 0 at start -> 1 at release, then held
    q_new = np.clip(ref.q + w * dq, kin.lo, kin.hi)
    p_new, n_new = kin.fk(q_new[k_rel])
    P2 = np.array([kin.fk(q)[0] for q in q_new])
    V2 = np.gradient(P2, dt, axis=0)
    info = {
        "source": path.name, "release_t_original": ref.release_t, "release_t": float(ref.t[k_rel]),
        "hand_velocity_at_release_original": V[k_rel].round(3).tolist(),
        "hand_velocity_at_release": V2[k_rel].round(3).tolist(),
        "disc_normal_z_before": float(kin.fk(q_rel)[1][2]), "disc_normal_z_after": float(n_new[2]),
        "hand_shift_m": float(np.linalg.norm(p_new - p_rel)), "joint_offset_rad": dq.round(3).tolist(),
        "peak_joint_speed_rad_s": float(np.abs(np.gradient(q_new, dt, axis=0)).max()),
    }
    return ref, q_new, k_rel, info


def main():
    kin = Kin()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for f in SOURCE_FILES:
        ref, q, k_rel, info = adapt(DATA_DIR / f, kin)
        out = OUT_DIR / f
        hdr = (f"t,joint_ra0..joint_ra7 (s, rad): {f} adapted for a flat frisbee grip\n"
               f"release at t={ref.t[k_rel]:.3f}s here (peak forward hand speed; original {info['release_t_original']:.3f}s)\n"
               f"disc normal z at release {info['disc_normal_z_before']:+.2f} -> {info['disc_normal_z_after']:+.2f}, "
               f"hand moved {info['hand_shift_m'] * 100:.1f} cm")
        np.savetxt(out, np.c_[ref.t, q], delimiter=",", fmt="%.6f", header=hdr, comments="# ")
        summary.append(info)
        print(f"{f}: release {info['release_t_original']:.2f}s -> {info['release_t']:.2f}s, "
              f"hand v {info['hand_velocity_at_release']}, disc n_z {info['disc_normal_z_before']:+.2f} -> "
              f"{info['disc_normal_z_after']:+.2f}, hand moved {info['hand_shift_m'] * 100:.1f} cm, "
              f"peak joint speed {info['peak_joint_speed_rad_s']:.2f} rad/s")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
