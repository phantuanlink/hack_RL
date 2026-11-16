"""Enlarge the (adapted) human throws and add a wrist flick: longer run-up, faster hand, real spin.

    uv run python -m hack_rl.enlarge_references [--max-scale 2.0] [--flick 1.0] [--flick-time 0.3]

1. Enlarge: every joint's motion is scaled about the release pose, q_s(t) = q_rel + s_j (q(t) - q_rel).
   The release pose (disc flat, from prepare_references) is unchanged; wind-up and follow-through grow,
   so the same playback speed moves the hand faster and over a longer path. s_j is as large as the joint
   limits allow (<= --max-scale); the global scale is reduced until no frame hits the torso, head or the
   parked left arm.
2. Flick: joint_ra7 turns about the disc normal, so its speed is spin. The wrist is cocked back by A
   during the wind-up, then snaps from -A to +A through the release (sinusoid, peak speed A*pi/(2*T)
   at release), in the direction the arm already spins the disc.

Writes data/reference/enlarged/<name>.csv (same format, "release at t=..s").
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from hack_rl.reference import DATA_DIR, DEFAULT_FILES, load_reference
from hack_rl.robot import panther as P

ARM_BODIES = ("link_ra3", "link_ra4", "link_ra5", "link_ra6", "link_ra7", "gripper", "finger_l", "finger_r")


def disc_state(m, d, gb, q, adr):
    d.qpos[adr] = q
    mujoco.mj_kinematics(m, d)
    R = d.xmat[gb].reshape(3, 3)
    Rrel = np.zeros(9)
    mujoco.mju_quat2Mat(Rrel, np.array(P.GRIP_REL_QUAT))
    Rd = R @ Rrel.reshape(3, 3)
    return d.xpos[gb] + R @ np.array(P.GRIP_REL_POS), Rd[:, 2]


BODY_GEOMS = ("torso_box", "head")      # the parked left arm is moved away for throws (it can be; the torso can't)


def min_gap(m, d, qs, adr, arm_ids):
    body = [m.geom(n).id for n in BODY_GEOMS]
    arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_ids]
    best = 1e9
    for q in qs:
        d.qpos[adr] = q
        mujoco.mj_forward(m, d)
        best = min(best, min(_gap(m, d, b, g) for b in body for g in arm))
    return best


MARGIN = 0.03


def hits_body(m, d, q, adr, arm_ids, torso_id, margin=None):
    """True if the arm comes within `margin` of the torso box or the head (capsule/box distances)."""
    margin = MARGIN if margin is None else margin
    d.qpos[adr] = q
    mujoco.mj_forward(m, d)
    body = [m.geom(n).id for n in BODY_GEOMS]
    arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_ids]
    for gb_ in body:
        for ga in arm:
            if _gap(m, d, gb_, ga) < margin:
                return True
    return False


def _gap(m, d, gbox, g):
    """Distance from geom g to the box gbox, sampling g's surface points (mj_geomDistance is unreliable for boxes)."""
    Rb = d.geom_xmat[gbox].reshape(3, 3); cb = d.geom_xpos[gbox]; sb = m.geom_size[gbox]
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; sz = m.geom_size[g]
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        pts = [c + R @ (np.array(sg) * sz) for sg in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T]
        rad = 0.0
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        pts = [c + R @ np.array([0, 0, h * sz[1]]) for h in np.linspace(-1, 1, 7)]
        rad = sz[0]
    else:
        pts, rad = [c], (sz[0] if t != mujoco.mjtGeom.mjGEOM_SPHERE else sz[0])
    if m.geom_type[gbox] == mujoco.mjtGeom.mjGEOM_SPHERE:
        return min(np.linalg.norm(p - cb) - sb[0] for p in pts) - rad
    best = 1e9
    for p in pts:
        lp = Rb.T @ (p - cb)
        out = np.maximum(np.abs(lp) - sb, 0)
        best = min(best, np.linalg.norm(out) if out.any() else -np.min(sb - np.abs(lp)))
    return best - rad


def enlarge(ref, m, d, adr, lo, hi, max_scale, arm_ids, torso_id):
    global MARGIN
    # never closer to the torso/head than the human throw already is (it passes ~1.5 cm from the crude torso
    # box), and never closer than 1 cm
    MARGIN = max(0.01, min(0.03, min_gap(m, d, ref.q[::2], adr, arm_ids) - 0.002))
    k = int(round(ref.release_t / (ref.t[1] - ref.t[0])))
    q_rel = ref.q[k]
    dq = ref.q - q_rel
    s_lim = np.full(8, max_scale)
    for j in range(8):
        up, dn = dq[:, j].max(), dq[:, j].min()
        if up > 1e-6:
            s_lim[j] = min(s_lim[j], (hi[j] - q_rel[j]) / up)
        if dn < -1e-6:
            s_lim[j] = min(s_lim[j], (lo[j] - q_rel[j]) / dn)
    s_lim = np.maximum(s_lim, 1.0)
    dt = ref.t[1] - ref.t[0]
    gb = m.body(P.GRIPPER_BODY).id

    def release_speed(qq):                    # disc-centre speed over the last 5 samples before release
        pts = np.array([disc_state(m, d, gb, x, adr)[0] for x in qq[k - 5:k + 1]])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean() / dt)

    def ok(qq):
        return not any(hits_body(m, d, qi, adr, arm_ids, torso_id) for qi in qq[::3])

    # greedy per joint: enlarge whichever joint speeds the disc up at release without nearing the body
    s = np.ones(8)
    best = release_speed(q_rel + s * dq)
    for _ in range(4):
        improved = False
        for j in range(8):
            for step in (0.4, 0.2, 0.1):
                if s[j] + step > s_lim[j] + 1e-9:
                    continue
                trial = s.copy()
                trial[j] += step
                qq = q_rel + trial * dq
                v = release_speed(qq)
                if v > best * 1.01 and ok(qq):
                    s, best, improved = trial, v, True
                    break
        if not improved:
            break
    return q_rel + s * dq, s, k


def add_flick(q, t, k, amp, T, sign, lo, hi):
    """ra7: cock to -sign*A over the 0.6 s before the flick, snap -A..+A through the release, hold."""
    q = q.copy()
    tr = t[k]
    A = min(amp, (hi[7] - q[:, 7].max()) if sign > 0 else (q[:, 7].min() - lo[7]),
            (q[:, 7].min() - lo[7]) if sign > 0 else (hi[7] - q[:, 7].max()))
    A = max(A, 0.0)
    delta = np.zeros_like(t)
    t0, t1 = tr - T, tr + T
    cock = (t > t0 - 0.6) & (t <= t0)
    x = (t[cock] - (t0 - 0.6)) / 0.6
    delta[cock] = -A * (3 * x ** 2 - 2 * x ** 3)
    snap = (t > t0) & (t <= t1)
    delta[snap] = A * np.sin(0.5 * math.pi * (t[snap] - tr) / T)
    delta[t > t1] = A
    q[:, 7] += sign * delta
    return q, A


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-scale", type=float, default=2.0)
    ap.add_argument("--flick", type=float, default=1.0, help="wrist flick amplitude A (rad, each side)")
    ap.add_argument("--flick-time", type=float, default=0.3, help="half duration T of the snap (s)")
    a = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML))
    d = mujoco.MjData(m)
    adr = np.array([m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)])
    rng = np.array([m.jnt_range[m.joint(f"joint_ra{i}").id] for i in range(8)])
    lo, hi = rng[:, 0] * 0.93, rng[:, 1] * 0.93
    gb = m.body(P.GRIPPER_BODY).id
    arm_ids = {m.body(b).id for b in ARM_BODIES}
    torso_id = m.body("torso").id
    out_dir = DATA_DIR / "enlarged"
    out_dir.mkdir(parents=True, exist_ok=True)
    ax7_local = m.jnt_axis[m.joint("joint_ra7").id]
    b7 = m.body("link_ra7").id
    for f in DEFAULT_FILES:
        ref = load_reference(DATA_DIR / f)
        dt = ref.t[1] - ref.t[0]
        q, s, k = enlarge(ref, m, d, adr, lo, hi, a.max_scale, arm_ids, torso_id)
        # spin direction the arm already gives the disc at release (about its normal)
        p0, n0 = disc_state(m, d, gb, ref.q[k - 1], adr)
        _, n1 = disc_state(m, d, gb, ref.q[k + 1], adr)
        d.qpos[adr] = q[k]
        mujoco.mj_kinematics(m, d)
        ax7 = d.xmat[b7].reshape(3, 3) @ ax7_local
        _, n_rel = disc_state(m, d, gb, q[k], adr)
        # angular velocity of the disc from the (enlarged) reference around the release
        R0 = np.zeros(9); R1 = np.zeros(9)
        def disc_R(qq):
            d.qpos[adr] = qq; mujoco.mj_kinematics(m, d)
            Rrel = np.zeros(9); mujoco.mju_quat2Mat(Rrel, np.array(P.GRIP_REL_QUAT))
            return d.xmat[gb].reshape(3, 3) @ Rrel.reshape(3, 3)
        Ra, Rb = disc_R(q[k - 1]), disc_R(q[k + 1])
        dR = Rb @ Ra.T
        w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * 2 * dt)
        spin_ref = float(w @ n_rel)
        sign = float(np.sign(spin_ref * (ax7 @ n_rel)) or np.sign(ax7 @ n_rel) or 1.0)
        q_f, A = add_flick(q, ref.t, k, a.flick, a.flick_time, sign, lo, hi)
        if any(hits_body(m, d, qi, adr, arm_ids, torso_id) for qi in q_f[::2]):
            print(f"{ref.name}: flick collides, using half amplitude")
            q_f, A = add_flick(q, ref.t, k, 0.5 * a.flick, a.flick_time, sign, lo, hi)
        # path length of the disc centre from the start to the release, before and after
        def path(qq):
            pts = np.array([disc_state(m, d, gb, x, adr)[0] for x in qq[:k + 1]])
            return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), float(np.linalg.norm(np.diff(pts[-6:], axis=0), axis=1).sum() / (5 * dt))
        L0, v0 = path(ref.q)
        L1, v1 = path(q_f)
        v7 = np.abs(np.gradient(q_f[:, 7], dt)).max()
        hdr = (f"t,joint_ra0..joint_ra7 (s, rad): {ref.name} enlarged about its release pose (per-joint scale "
               f"{np.round(s, 2).tolist()}) + wrist flick on joint_ra7 (A={A:.2f} rad, T={a.flick_time}s, sign {sign:+.0f})\n"
               f"release at t={ref.release_t:.3f}s here (as the adapted reference)\n"
               f"disc path start->release {L0:.2f} -> {L1:.2f} m; disc speed at release {v0:.2f} -> {v1:.2f} m/s at 1x; "
               f"peak ra7 speed {v7:.1f} rad/s; arm spin about the disc normal at release {spin_ref:+.1f} rad/s")
        np.savetxt(out_dir / f"{ref.name}.csv", np.c_[ref.t, q_f], delimiter=",", fmt="%.6f", header=hdr, comments="# ")
        print(f"{ref.name}: body margin {MARGIN * 100:.1f} cm; scale {np.round(s, 2).tolist()}, flick A={A:.2f} rad sign {sign:+.0f}; disc path {L0:.2f} -> {L1:.2f} m, "
              f"speed at release {v0:.2f} -> {v1:.2f} m/s (1x playback), peak ra7 {v7:.1f} rad/s")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
