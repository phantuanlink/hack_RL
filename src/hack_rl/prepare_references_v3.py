"""Reference preprocessing v3: retarget the tracked human arm by IK (positions, not copied angles), away from
the straight-arm singularity, at the human's timing.

    uv run python -m hack_rl.prepare_references_v3 [--reach-frac 0.93] [--flick 0.4]

Why: HEADBands' retarget copies the human's joint angles. At the whip the human elbow is nearly straight, so
the robot's elbow (ra4, straight = 0 = the end of its range) sits at the singularity: the arm is fully
extended and can't move the wrist along itself.

v3:
  1. human elbow + wrist positions relative to the shoulder (phone tracking, 60 Hz, HEADBands selections crop),
     Savitzky-Golay smoothed, at the human's own timing;
  2. scaled to the robot's arm (0.274 + 0.276 m vs the human's ~0.63 m), then the shoulder-wrist distance is
     softly compressed so it never exceeds --reach-frac of the robot's reach (0.93 -> elbow bent >= ~45 deg);
  3. IK (ra0-ra4) per frame, warm-started: wrist centre on the target (weight 1), elbow towards the scaled
     human elbow (weight 0.3: keeps the human's arm posture / swivel), a barrier keeping the elbow bent,
     joint limits, small step to the previous frame (smoothness);
  4. release = peak horizontal disc speed; wrist ra5-ra7 offset (blended in) makes the disc flat at the
     release; ra7 flick seed; human follow-through after the release, then a decelerating tail.
Writes data/reference/ik/<name>.csv + .json (release state, for render_throw.py) and prints the margins.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.signal import savgol_filter

from hack_rl.reference import DATA_DIR
from hack_rl.robot import panther as P

HUMAN_DIR = Path(__file__).resolve().parents[2] / "data" / "reference" / "human"
OUT = DATA_DIR / "ik"
SOURCES = ("trajectory_shoulder_rel_2.csv", "trajectory_shoulder_rel_3.csv")
CROPS = {"trajectory_shoulder_rel_2.csv": (1.017140, 1.917558), "trajectory_shoulder_rel_3.csv": (0.983871, 1.717605)}
ARM, WRIST = [0, 1, 2, 3, 4], [5, 6, 7]


class Robot:
    def __init__(self):
        m = self.m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML))
        self.d = mujoco.MjData(m)
        self.adr = np.array([m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)])
        r = np.array([m.jnt_range[m.joint(f"joint_ra{i}").id] for i in range(8)])
        self.lo, self.hi = r[:, 0] * 0.97, r[:, 1] * 0.97
        self.j = {n: m.joint(f"joint_ra{n}").id for n in range(8)}
        self.gb = m.body(P.GRIPPER_BODY).id
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.array(P.GRIP_REL_QUAT))
        self.R_rel = R.reshape(3, 3)
        q0 = np.zeros(8)
        S, E, W = self.points(q0)
        tb, hd = m.geom("torso_box").id, m.geom("head").id
        self.torso_c, self.torso_R, self.torso_s = self.d.geom_xpos[tb].copy(), self.d.geom_xmat[tb].reshape(3, 3).copy(), m.geom_size[tb].copy()
        self.head_c, self.head_r = self.d.geom_xpos[hd].copy(), m.geom_size[hd][0]
        self.S = S
        self.L1, self.L2 = np.linalg.norm(E - S), np.linalg.norm(W - E)

    def points(self, q):
        """shoulder centre, elbow, wrist centre (world/torso frame)."""
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        # anchors are only filled by mj_forward-level calls; compute from body frames instead
        mujoco.mj_forward(self.m, self.d)
        a = self.d.xanchor
        return a[self.j[2]].copy(), a[self.j[4]].copy(), a[self.j[6]].copy()

    def body_points(self):
        """after points(): elbow, mid-forearm, wrist centre, finger tips (the parts that can hit the torso)."""
        a = self.d.xanchor
        E, W = a[self.j[4]].copy(), a[self.j[6]].copy()
        return [E, 0.5 * (E + W), W, self.d.site_xpos[self.m.site("grip_site").id].copy()]

    def body_clearance(self, pts):
        """signed distance of each point to the torso box / head sphere (negative = inside)."""
        out = []
        for p in pts:
            lp = self.torso_R.T @ (p - self.torso_c)
            o = np.maximum(np.abs(lp) - self.torso_s, 0)
            db = np.linalg.norm(o) if o.any() else -np.min(self.torso_s - np.abs(lp))
            dh = np.linalg.norm(p - self.head_c) - self.head_r
            out.append(min(db, dh))
        return np.array(out)

    def disc(self, q):
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        Rg = self.d.xmat[self.gb].reshape(3, 3)
        return self.d.xpos[self.gb] + Rg @ np.array(P.GRIP_REL_POS), Rg @ self.R_rel

    def elbow_angle(self, q):
        S, E, W = self.points(q)
        u, v = S - E, W - E
        return math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))  # 180 = straight


def soft_clip(r, r_max, knee=0.85):
    """identity below knee*r_max, smooth saturation to r_max above."""
    r0 = knee * r_max
    out = r.copy()
    hi = r > r0
    span = r_max - r0
    out[hi] = r0 + span * np.tanh((r[hi] - r0) / span)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reach-frac", type=float, default=0.93)
    ap.add_argument("--min-bend-deg", type=float, default=35.0, help="IK barrier: elbow at least this far from straight")
    ap.add_argument("--flick", type=float, default=0.4)
    ap.add_argument("--flick-time", type=float, default=0.08)
    ap.add_argument("--tail", type=float, default=0.3)
    ap.add_argument("--body-margin", type=float, default=0.10, help="m: elbow/forearm/wrist/fingers off the torso and head")
    ap.add_argument("--ik-max-speed", type=float, default=12.0,
                    help="rad/s per joint between IK frames: stay on one continuous solution (no branch jumps)")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--chest-frame", action="store_true",
                    help="v4: express the arm in the human's CHEST frame (rotating with the torso twist), from the left/"
                         "right shoulder keypoints; the robot torso is fixed, so the twist must not be in the arm path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stroke-only", action="store_true",
                    help="drop the human wind-up: start at the deepest backswing (along the release direction), hold "
                         "--hold s there, then the back-to-front stroke, release, follow-through")
    ap.add_argument("--hold", type=float, default=0.3)
    a = ap.parse_args(argv)
    rb = Robot()
    reach = rb.L1 + rb.L2
    out_dir = Path(a.out) if a.out else (DATA_DIR / "chest" if a.chest_frame else OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"robot: upper arm {rb.L1:.3f} m, forearm {rb.L2:.3f} m, reach {reach:.3f} m, shoulder at {rb.S.round(3)}")
    for src in SOURCES:
        raw = np.loadtxt(HUMAN_DIR / src, delimiter=",", skiprows=1)
        t0, t1 = CROPS[src]
        keep = (raw[:, 0] >= t0) & (raw[:, 0] <= t1)
        th = raw[keep, 0] - t0
        E_raw, W_raw = raw[keep, 1:4], raw[keep, 4:7]
        chest_yaw = None
        if a.chest_frame:                                       # rotate each frame by -(chest yaw) about z
            doc = json.loads((HUMAN_DIR / f"trajectory_throw{src.split('_')[-1].split('.')[0]}.json").read_text())
            tj = np.array(doc["timestamp"], float)
            Ls = np.array(doc["extra_joints"]["left_shoulder"], float)
            Rs = np.array(doc["extra_joints"]["right_shoulder"], float)
            left = Ls - Rs
            yaw_all = np.unwrap(np.arctan2(-left[:, 0], left[:, 1]))   # forward = left x up = (l_y, -l_x)
            yaw_all = savgol_filter(yaw_all, 11, 2)
            chest_yaw = np.interp(raw[keep, 0], tj, yaw_all)
            c, s_ = np.cos(-chest_yaw), np.sin(-chest_yaw)
            def rot(P_):
                return np.stack([c * P_[:, 0] - s_ * P_[:, 1], s_ * P_[:, 0] + c * P_[:, 1], P_[:, 2]], 1)
            E_raw, W_raw = rot(E_raw), rot(W_raw)
        E_h = savgol_filter(E_raw, 7, 2, axis=0)
        W_h = savgol_filter(W_raw, 7, 2, axis=0)
        h_reach = np.percentile(np.linalg.norm(E_h, axis=1) + np.linalg.norm(W_h - E_h, axis=1), 50)
        s = reach / h_reach
        t = np.arange(0.0, th[-1] + 1e-9, a.dt)
        Wt = np.stack([np.interp(t, th, W_h[:, i]) for i in range(3)], 1) * s
        Et = np.stack([np.interp(t, th, E_h[:, i]) for i in range(3)], 1) * s
        r = np.linalg.norm(Wt, axis=1)
        r_new = soft_clip(r, a.reach_frac * reach)
        Wt = Wt * (r_new / np.maximum(r, 1e-9))[:, None] + rb.S
        Et = Et + rb.S
        # IK per frame
        q = np.zeros(8)
        q[4] = -1.0                                              # start bent
        Q = []
        cos_max = math.cos(math.radians(180 - a.min_bend_deg))   # interior angle <= 180 - min_bend

        def res(x, w_t, e_t, q_prev):
            qq = q.copy()
            qq[ARM] = x
            S_, E_, W_ = rb.points(qq)
            u, v = S_ - E_, W_ - E_
            c = u @ v / np.linalg.norm(u) / np.linalg.norm(v)    # -1 = straight
            barrier = max(0.0, cos_max - c) * 5.0                # penalty when straighter than allowed
            clear = np.maximum(0.0, a.body_margin - rb.body_clearance(rb.body_points())) * 10.0   # keep off the torso
            return np.r_[W_ - w_t, 0.3 * (E_ - e_t), barrier, clear, 0.05 * (x - q_prev[ARM])]

        step = a.ik_max_speed * a.dt
        for i in range(len(t)):
            starts = [q[ARM]] if i else [q[ARM]] + [np.random.default_rng(k).uniform(rb.lo[ARM], rb.hi[ARM]) for k in range(40)]
            if i:                                                # continuity: at most ik_max_speed from the last frame
                lo_i = np.maximum(rb.lo[ARM], q[ARM] - step)
                hi_i = np.minimum(rb.hi[ARM], q[ARM] + step)
            else:
                lo_i, hi_i = rb.lo[ARM], rb.hi[ARM]
            best = None
            for x0 in starts:
                sol = least_squares(res, np.clip(x0, lo_i + 1e-9, hi_i - 1e-9), args=(Wt[i], Et[i], q),
                                    bounds=(lo_i, hi_i), max_nfev=200)
                if best is None or sol.cost < best.cost:
                    best = sol
            q = q.copy()
            q[ARM] = best.x
            Q.append(q.copy())
        Q = np.array(Q)
        Q[:, ARM] = savgol_filter(Q[:, ARM], 11, 3, axis=0)    # remove IK step noise
        err = np.array([np.linalg.norm(rb.points(x)[2] - w) for x, w in zip(Q, Wt)])
        clr = np.array([(rb.points(x), rb.body_clearance(rb.body_points()).min())[1] for x in Q])
        bend = np.array([180 - rb.elbow_angle(x) for x in Q])
        # release at peak horizontal disc speed (wrist joints at 0 for now)
        Pd = np.array([rb.disc(x)[0] for x in Q])
        sp = np.linalg.norm(np.gradient(Pd, a.dt, axis=0)[:, :2], axis=1)
        k = int(np.argmax(sp[: int(0.95 * len(sp))]))
        b = 0
        if a.stroke_only:
            d_rel = np.gradient(Pd, a.dt, axis=0)[k, :2]
            d_rel = d_rel / np.linalg.norm(d_rel)
            b = int(np.argmin(Pd[:k + 1, :2] @ d_rel))          # deepest backswing along the throw direction
            n_hold = int(round(a.hold / a.dt))
            Q = np.vstack([np.repeat(Q[b:b + 1], n_hold, 0), Q[b:]])
            Wt = np.vstack([np.repeat(Wt[b:b + 1], n_hold, 0), Wt[b:]])
            err = np.r_[np.repeat(err[b:b + 1], n_hold), err[b:]]
            bend = np.r_[np.repeat(bend[b:b + 1], n_hold), bend[b:]]
            clr = np.r_[np.repeat(clr[b:b + 1], n_hold), clr[b:]]
            k = k - b + n_hold
            t = np.arange(len(Q)) * a.dt
        # wrist-only flattening at the release
        q_rel = Q[k].copy()

        def cost(x):
            qq = q_rel.copy()
            qq[WRIST] += x
            return (1 - rb.disc(qq)[1][2, 2]) ** 2 + 0.01 * (x @ x)
        bounds = [(rb.lo[j] - q_rel[j], rb.hi[j] - q_rel[j]) for j in WRIST]
        best, rng = None, np.random.default_rng(0)
        for trial in range(40):
            x0 = np.zeros(3) if trial == 0 else rng.uniform(-1.5, 1.5, 3)
            r_ = minimize(cost, np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]), method="L-BFGS-B", bounds=bounds)
            if best is None or r_.fun < best.fun:
                best = r_
        wgt = (lambda x: x * x * (3 - 2 * x))(np.clip(t / t[k], 0, 1))[:, None]
        off = np.zeros(8)
        off[WRIST] = best.x
        Q = np.clip(Q + wgt * off, rb.lo, rb.hi)
        # spin direction of the arm at release, ra7 flick seed in that direction
        Ra, Rb2 = rb.disc(Q[k - 1])[1], rb.disc(Q[k + 1])[1]
        dR = Rb2 @ Ra.T
        w_arm = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * 2 * a.dt)
        n_rel = rb.disc(Q[k])[1][:, 2]
        rb.d.qpos[rb.adr] = Q[k]
        mujoco.mj_forward(rb.m, rb.d)
        ax7 = rb.d.xaxis[rb.j[7]].copy()
        sign = float(np.sign((w_arm @ n_rel) * (ax7 @ n_rel)) or 1.0)
        A = max(0.0, min(a.flick, rb.hi[7] - Q[:, 7].max(), Q[:, 7].min() - rb.lo[7]))
        T = a.flick_time
        delta = np.zeros_like(t)
        cock = (t > t[k] - T - 0.15) & (t <= t[k] - T)
        xx = (t[cock] - (t[k] - T - 0.15)) / 0.15
        delta[cock] = -A * (3 * xx ** 2 - 2 * xx ** 3)
        snap = t > t[k] - T
        delta[snap] = A * np.sin(0.5 * math.pi * np.clip((t[snap] - t[k]) / T, -1, 1))
        Q[:, 7] += sign * delta
        # follow-through tail
        v_end = (Q[-1] - Q[-2]) / a.dt
        n = int(round(a.tail / a.dt))
        tail = [Q[-1]]
        for i in range(1, n + 1):
            tail.append(np.clip(tail[-1] + v_end * (1 - i / n) * a.dt, rb.lo, rb.hi))
        Qf = np.vstack([Q, np.array(tail[1:])])
        tf = np.arange(len(Qf)) * a.dt
        # report + release state
        Pf = np.array([rb.disc(x)[0] for x in Qf])
        V = np.gradient(Pf, a.dt, axis=0)
        Ra, Rb2 = rb.disc(Qf[k - 1])[1], rb.disc(Qf[k + 1])[1]
        dR = Rb2 @ Ra.T
        w_rel = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * 2 * a.dt)
        n_rel = rb.disc(Qf[k])[1][:, 2]
        v = V[k]
        heading = math.degrees(math.atan2(v[1], v[0]))
        L = float(np.linalg.norm(np.diff(Pf[:k + 1], axis=0), axis=1).sum())
        if a.stroke_only:
            hdr_extra = f"; starts at the deepest backswing (human wind-up dropped), {a.hold:.2f}s hold"
        else:
            hdr_extra = ""
        jv = np.abs(np.gradient(Qf, a.dt, axis=0)).max(0)
        name = "joints_" + Path(src).stem
        hdr = (f"t,joint_ra0..joint_ra7 (s, rad): {Path(src).stem} retargeted by IK (v3): human timing, scale {s:.2f}, "
               f"shoulder-wrist <= {a.reach_frac:.2f} reach, elbow bent >= {bend.min():.0f} deg (min), wrist flattened "
               f"{np.round(best.x, 2).tolist()}, ra7 flick {A:.2f} rad\n"
               f"release at t={t[k]:.3f}s here (peak horizontal disc speed)\n"
               f"disc path to release {L:.2f} m, disc velocity at release {np.round(v, 2).tolist()} m/s "
               f"(heading {heading:+.0f} deg), spin about the normal {w_rel @ n_rel:+.1f} rad/s, flat {n_rel[2]:.2f}; "
               f"IK wrist error mean {err.mean() * 100:.1f} cm, max {err.max() * 100:.1f} cm{hdr_extra}")
        if chest_yaw is not None:
            cy = np.degrees(np.interp(t[k], th, chest_yaw))
            hdr = hdr.replace("retargeted by IK (v3)", f"retargeted by IK in the human CHEST frame (v4; chest yaw at "
                              f"release {cy:+.0f} deg, twist {np.degrees(chest_yaw.max() - chest_yaw.min()):.0f} deg removed)")
        np.savetxt(out_dir / f"{name}.csv", np.c_[tf, Qf], delimiter=",", fmt="%.6f", header=hdr, comments="# ")
        (out_dir / f"{name}.json").write_text(json.dumps({"frisbee_leaves_t": float(t[k]), "velocity_mps": v.tolist(),
                                                       "angular_velocity_radps": w_rel.tolist()}))
        print(f"{name}: scale {s:.2f}, IK wrist error mean {err.mean() * 100:.1f} cm (max {err.max() * 100:.1f}); "
              f"elbow bend min {bend.min():.0f} deg (straight = 0), ra4 closest to straight {Qf[:, 4].max():+.2f} rad; "
              f"min clearance to torso/head (joint centres) {clr.min() * 100:.1f} cm; "
              f"release t={t[k]:.2f}s: disc {np.linalg.norm(v[:2]):.1f} m/s heading {heading:+.0f} deg, spin {w_rel @ n_rel:+.1f} rad/s, "
              f"flat {n_rel[2]:.2f}; path to release {L:.2f} m\n    peak joint speed (rad/s) {jv.round(1)}")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
