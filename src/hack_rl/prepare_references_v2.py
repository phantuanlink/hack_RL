"""Reference preprocessing v2: the human throw at the human's own timing, the full swing, wrist-only flattening.

    uv run python -m hack_rl.prepare_references_v2 [--flick 0.4] [--flick-time 0.08] [--tail 0.3]

Input: HEADBands example_nathan joints_trajectory_shoulder_rel_*.csv (data/reference/). Their retarget copies the
human's joint angles (ra0 yaw, ra1-ra3 spherical shoulder, ra4 elbow) from the phone tracking, cropped from the
start of the throw to the human release, then *slows it down* (e.g. 6.4x) so no joint exceeds 3 rad/s, sets
ra5 = 0 and adds a +-20 deg wrist flick.

v2 (v1 = prepare_references.py, which moved the release to mid-swing and tilted all 8 joints):
  1. undo the slow-down: time back to the human's (the crop of ref3 is 0.73 s);
  2. release = the moment of peak horizontal hand speed (the whip; ~7-9 m/s, heading -15..-30 deg). The crop's
     "release" is the END of the follow-through (the hand is already slowing and turning back there); the
     human follow-through after the peak is kept;
  3. the arm (ra0-ra4, what the phones measured) is left exactly as the human did it;
  4. the wrist (ra5-ra7, not measured) gets an offset, blended in over the swing, that makes the disc flat at
     the release; plus an optional flick seed on ra7 (its axis is the disc normal: its speed is spin);
  5. after the release a follow-through: every joint keeps its release speed, decaying to zero over --tail s.
The policy then plays it at 0.5-1.5x and adds residuals; the joint limiter clips what the motors can't do.
Writes data/reference/human_speed/<name>.csv (+ prints swing length, disc speed and heading at release).
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

from hack_rl.reference import DATA_DIR
from hack_rl.robot import panther as P

SOURCES = ("joints_trajectory_shoulder_rel_2.csv", "joints_trajectory_shoulder_rel_3.csv")
OUT = DATA_DIR / "human_speed"
WRIST = [5, 6, 7]


class Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML))
        self.d = mujoco.MjData(self.m)
        self.adr = np.array([self.m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)])
        r = np.array([self.m.jnt_range[self.m.joint(f"joint_ra{i}").id] for i in range(8)])
        self.lo, self.hi = r[:, 0] * 0.95, r[:, 1] * 0.95
        self.gb = self.m.body(P.GRIPPER_BODY).id
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.array(P.GRIP_REL_QUAT))
        self.R_rel = R.reshape(3, 3)

    def disc(self, q):
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        Rg = self.d.xmat[self.gb].reshape(3, 3)
        return self.d.xpos[self.gb] + Rg @ np.array(P.GRIP_REL_POS), Rg @ self.R_rel


def load_raw(path: Path):
    text = path.read_text()
    rows = np.loadtxt(path, delimiter=",", comments="#")
    stretch = float(re.search(r"played ([\d.]+)x slower", text).group(1))
    t_rel = float(re.search(r"release at t=([\d.]+)s", text).group(1))
    return rows[:, 0], rows[:, 1:9], stretch, t_rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flick", type=float, default=0.4, help="ra7 flick seed amplitude each side (rad); 0 = none")
    ap.add_argument("--flick-time", type=float, default=0.08, help="half duration of the snap (human s)")
    ap.add_argument("--tail", type=float, default=0.3, help="follow-through after the release (s)")
    ap.add_argument("--dt", type=float, default=0.005)
    a = ap.parse_args(argv)
    kin = Kin()
    OUT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        t_raw, q_raw, stretch, t_rel_raw = load_raw(DATA_DIR / src)
        keep = t_raw <= t_rel_raw + 1e-9                          # drop HEADBands' robot-time slow-down tail
        th = t_raw[keep] / stretch                                # human time
        t = np.arange(0.0, th[-1] + 1e-9, a.dt)
        q = np.stack([np.interp(t, th, q_raw[keep, j]) for j in range(8)], 1)
        Pc = np.array([kin.disc(x)[0] for x in q])
        sp = np.linalg.norm(np.gradient(Pc, a.dt, axis=0)[:, :2], axis=1)
        k = int(np.argmax(sp))                                    # release = peak horizontal hand speed (the whip)
        # wrist-only flattening at the release
        q_rel = q[k]

        def cost(x):
            qq = q_rel.copy()
            qq[WRIST] += x
            _, R = kin.disc(qq)
            return (1 - R[2, 2]) ** 2 + 0.01 * (x @ x)
        bounds = [(kin.lo[j] - q_rel[j], kin.hi[j] - q_rel[j]) for j in WRIST]
        best, rng = None, np.random.default_rng(0)
        for trial in range(40):
            x0 = np.zeros(3) if trial == 0 else rng.uniform(-1.5, 1.5, 3)
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            r = minimize(cost, x0, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        w = (lambda x: x * x * (3 - 2 * x))(np.clip(t / t[k], 0, 1))[:, None]       # held after the release
        off = np.zeros(8)
        off[WRIST] = best.x
        q = np.clip(q + w * off, kin.lo, kin.hi)
        # disc velocity at release (human timing), for the flick direction and the report
        P0 = np.array([kin.disc(x)[0] for x in q])
        V = np.gradient(P0, a.dt, axis=0)
        Ra, Rb = kin.disc(q[k - 1])[1], kin.disc(q[k])[1]
        dR = Rb @ Ra.T
        w_disc = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * a.dt)
        n_rel = Rb[:, 2]
        spin_arm = float(w_disc @ n_rel)
        # ra7 flick seed, in the direction the arm already spins the disc
        A = 0.0
        if a.flick > 0:
            kin.d.qpos[kin.adr] = q[k]
            mujoco.mj_kinematics(kin.m, kin.d)
            b7 = kin.m.body("link_ra7").id
            ax7 = kin.d.xmat[b7].reshape(3, 3) @ kin.m.jnt_axis[kin.m.joint("joint_ra7").id]
            sign = float(np.sign(spin_arm * (ax7 @ n_rel)) or 1.0)
            A = min(a.flick, kin.hi[7] - q[:, 7].max(), q[:, 7].min() - kin.lo[7])
            T = a.flick_time
            delta = np.zeros_like(t)
            cock = (t > t[k] - T - 0.15) & (t <= t[k] - T)
            x = (t[cock] - (t[k] - T - 0.15)) / 0.15
            delta[cock] = -A * (3 * x ** 2 - 2 * x ** 3)
            snap = t > t[k] - T
            delta[snap] = A * np.sin(0.5 * math.pi * np.clip((t[snap] - t[k]) / T, -1, 1))
            q[:, 7] += sign * delta
        # follow-through: the human's own (after the peak, to the end of the crop), then every joint decelerates
        # from its last speed to zero over --tail
        v_rel = (q[-1] - q[-2]) / a.dt
        n = int(round(a.tail / a.dt))
        tail = [q[-1]]
        for i in range(1, n + 1):
            tail.append(np.clip(tail[-1] + v_rel * (1 - i / n) * a.dt, kin.lo, kin.hi))
        q_full = np.vstack([q, np.array(tail[1:])])
        t_full = np.arange(len(q_full)) * a.dt
        # report
        P1 = np.array([kin.disc(x)[0] for x in q_full])
        V1 = np.gradient(P1, a.dt, axis=0)
        L = float(np.linalg.norm(np.diff(P1[:k + 1], axis=0), axis=1).sum())
        v = V1[k]
        heading = math.degrees(math.atan2(v[1], v[0]))
        jv = np.abs(np.gradient(q_full, a.dt, axis=0))
        name = Path(src).stem
        hdr = (f"t,joint_ra0..joint_ra7 (s, rad): {name} v2 - HUMAN timing (HEADBands' {stretch:.2f}x slow-down undone), "
               f"full swing, arm ra0-ra4 as tracked, wrist ra5-7 flattened (offset {np.round(best.x, 2).tolist()}), "
               f"ra7 flick seed {A:.2f} rad\n"
               f"release at t={t[k]:.3f}s here (peak horizontal hand speed; the crop ends at {t[-1]:.3f}s)\n"
               f"disc path to release {L:.2f} m; disc velocity at release {np.round(v, 2).tolist()} m/s "
               f"({np.linalg.norm(v[:2]):.1f} m/s horizontal, heading {heading:+.0f} deg); disc normal z {kin.disc(q[k])[1][2, 2]:.2f}")
        np.savetxt(OUT / f"{name}.csv", np.c_[t_full, q_full], delimiter=",", fmt="%.6f", header=hdr, comments="# ")
        print(f"{name}: human timing (was {stretch:.2f}x slower), swing {t[k]:.2f}s, disc path to release {L:.2f} m, "
              f"disc velocity at release {np.round(v, 2).tolist()} ({np.linalg.norm(v[:2]):.1f} m/s, heading {heading:+.0f} deg), "
              f"flat {kin.disc(q[k])[1][2, 2]:.2f}, arm spin {spin_arm:+.1f} rad/s, flick seed {A:.2f} rad\n"
              f"    peak joint speed (rad/s) {jv.max(0).round(1)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
