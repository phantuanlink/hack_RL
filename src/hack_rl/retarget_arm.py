"""Retarget the (adapted) Panther reference throws to another arm, e.g. the Franka Panda.

    uv run python -m hack_rl.retarget_arm --robot franka

What is kept: the trajectory of the frisbee (its centre and its normal) relative to the Panther shoulder,
which is the human throw. The new arm's shoulder is put where the Panther shoulder is, and its mounting
orientation is chosen by search (the one whose IK tracks the throw best). Then weighted damped-least-squares
IK frame by frame: disc centre (1 m/m), disc normal (0.3 m/rad) and, weakly, the rotation about the normal
(0.05 m/rad; the disc is symmetric there, but that rotation is the spin, so it is not ignored).

Writes data/reference/<robot>/<ref>.csv (same format as the Panther references) and mount.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from hack_rl.reference import DATA_DIR, load_reference

W_POS, W_TILT, W_SPIN = 1.0, 0.3, 0.05


def _quat_from_mat(m):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, m.reshape(-1))
    return q


def _rot(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), ang)
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def disc_trajectory(files):
    """Panther FK: disc centre and orientation (world, torso unrotated) for each reference, + shoulder."""
    from hack_rl.robot import panther as P
    m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML))
    d = mujoco.MjData(m)
    jids = [m.joint(j).qposadr[0] for j in P.ARM_JOINTS]
    gb = m.body(P.GRIPPER_BODY).id
    rel_p = np.array(P.GRIP_REL_POS)
    rel_R = np.zeros(9)
    mujoco.mju_quat2Mat(rel_R, np.array(P.GRIP_REL_QUAT))
    rel_R = rel_R.reshape(3, 3)
    mujoco.mj_forward(m, d)
    shoulder = d.xpos[m.body("link_ra1").id].copy() + [0, 0, P.TORSO_HEIGHT]
    out = []
    for f in files:
        ref = load_reference(DATA_DIR / f)
        pos, rot = [], []
        for q in ref.q:
            d.qpos[jids] = q
            mujoco.mj_kinematics(m, d)
            R = d.xmat[gb].reshape(3, 3)
            pos.append(d.xpos[gb] + R @ rel_p + [0, 0, P.TORSO_HEIGHT])
            rot.append(R @ rel_R)
        out.append((ref, np.array(pos), np.array(rot)))
    return shoulder, out


class ArmIK:
    def __init__(self, profile, mount_pos, mount_R):
        from hack_rl.robot.franka import ik_model
        self.p = profile
        self.m = ik_model(mount_pos, _quat_from_mat(mount_R))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in profile.arm_joints])
        self.dadr = np.array([self.m.joint(j).dofadr[0] for j in profile.arm_joints])
        self.lo = np.array([self.m.joint(j).range[0] for j in profile.arm_joints]) * 0.97
        self.hi = np.array([self.m.joint(j).range[1] for j in profile.arm_joints]) * 0.97
        self.gb = self.m.body(profile.gripper_body).id
        rel_R = np.zeros(9)
        mujoco.mju_quat2Mat(rel_R, np.array(profile.grip_rel_quat))
        self.rel_R = rel_R.reshape(3, 3)
        self.rel_p = np.array(profile.grip_rel_pos)

    def disc_pose(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.gb].reshape(3, 3)
        return self.d.xpos[self.gb] + R @ self.rel_p, R @ self.rel_R

    def error(self, q, p_t, R_t):
        p, R = self.disc_pose(q)
        e_rot = np.zeros(3)                       # rotation taking R to R_t, in world
        qt, qc = _quat_from_mat(R_t), _quat_from_mat(R)
        qc_inv = np.zeros(4)
        mujoco.mju_negQuat(qc_inv, qc)
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, qt, qc_inv)
        mujoco.mju_quat2Vel(e_rot, dq, 1.0)
        n = R[:, 2]
        e_spin = n * (e_rot @ n)
        return p_t - p, e_rot - e_spin, e_spin

    def solve(self, q, p_t, R_t, iters=40, lam=0.03):
        for _ in range(iters):
            e_p, e_tilt, e_spin = self.error(q, p_t, R_t)
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_tilt) < 1e-3:
                break
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            # Jacobian of the disc centre point (attached to the gripper body)
            p_disc, R_disc = self.disc_pose(q)
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jac(self.m, self.d, jp, jr, p_disc, self.gb)
            jp, jr = jp[:, self.dadr], jr[:, self.dadr]
            n = R_disc[:, 2]
            Pn = np.outer(n, n)
            J = np.vstack([W_POS * jp, W_TILT * (np.eye(3) - Pn) @ jr, W_SPIN * Pn @ jr])
            e = np.concatenate([W_POS * e_p, W_TILT * e_tilt, W_SPIN * e_spin])
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(9), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        e_p, e_tilt, e_spin = self.error(q, p_t, R_t)
        return q, np.linalg.norm(e_p), np.linalg.norm(e_tilt), np.linalg.norm(e_spin)

    def track(self, pos, rot, q0=None, starts=24, seed=0, dt=0.01, max_speed=None):
        rng = np.random.default_rng(seed)
        cands = [q0] if q0 is not None else []
        cands += [rng.uniform(self.lo, self.hi) for _ in range(starts)]
        best = min((self.solve(c, pos[0], rot[0], iters=200) for c in cands), key=lambda r: r[1] + 0.3 * r[2])
        q = best[0]
        qs, errs = [], []
        cap = None if max_speed is None else np.asarray(max_speed) * dt
        for p_t, R_t in zip(pos, rot):
            q_prev = q.copy()
            q, ep, et, es = self.solve(q, p_t, R_t)
            if cap is not None and np.any(np.abs(q - q_prev) > cap):
                # never jump to another IK branch: move at most max_speed, then re-solve from there
                q = q_prev + np.clip(q - q_prev, -cap, cap)
                ep, et, es = (np.linalg.norm(x) for x in self.error(q, p_t, R_t))
            qs.append(q.copy())
            errs.append((ep, et, es))
        return np.array(qs), np.array(errs)


def mount_candidates():
    """Joint-1 axis along +-x, +-y, +-z, times 4 rotations about it."""
    out = []
    for ax, sgn in itertools.product(range(3), (1, -1)):
        z = np.zeros(3)
        z[ax] = sgn
        # any perpendicular x
        x0 = np.array([1.0, 0, 0]) if ax != 0 else np.array([0, 0, 1.0])
        x0 = x0 - z * (x0 @ z)
        x0 /= np.linalg.norm(x0)
        for k in range(4):
            x = _rot(z, k * math.pi / 2) @ x0
            out.append((f"z{'+-'[sgn < 0]}{'xyz'[ax]}_r{90 * k}", np.c_[x, np.cross(z, x), z]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="franka")
    a = ap.parse_args(argv)
    from hack_rl.robot.profile import get_profile
    from hack_rl.reference import DEFAULT_FILES
    prof = get_profile(a.robot)
    shoulder, trajs = disc_trajectory(DEFAULT_FILES)
    reach = max(np.linalg.norm(p - shoulder, axis=1).max() for _, p, _ in trajs)
    print(f"Panther shoulder (world) {shoulder.round(3)}, farthest disc centre {reach:.3f} m from it")

    from hack_rl.robot.franka import SHOULDER_IN_BASE
    scores = []
    for name, R in mount_candidates():
        base = shoulder - R @ np.array(SHOULDER_IN_BASE)
        ik = ArmIK(prof, base, R)
        tot = 0.0
        for _, pos, rot in trajs:
            _, e = ik.track(pos[::10], rot[::10], starts=12)
            tot += (e[:, 0] + 0.3 * e[:, 1]).mean()
        scores.append((tot / len(trajs), name, R, base))
        print(f"  mount {name:>10}: mean error {tot / len(trajs):.4f}")
    scores.sort(key=lambda s: s[0])
    err, name, R, base = scores[0]
    print(f"best mount {name} (error {err:.4f}); base at {base.round(3)}")
    ik = ArmIK(prof, base, R)
    out_dir = DATA_DIR / a.robot
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"mount": name, "base_pos": base.tolist(), "base_quat": _quat_from_mat(R).tolist(),
               "shoulder_world": shoulder.tolist(), "references": {}}
    vmax = np.array(prof.vel_limits)
    for ref, pos, rot in trajs:
        dt = ref.t[1] - ref.t[0]
        # allow 2x the joint speed limit in the reference (the policy can slow down / the limiter caps it)
        q, e = ik.track(pos, rot, starts=48, dt=dt, max_speed=2.0 * np.array(prof.vel_limits))
        speed = np.abs(np.diff(q, axis=0)) / dt
        k = int(round(ref.release_t / dt))
        hdr = (f"t,{','.join(prof.arm_joints)} (s, rad): {ref.name} retargeted from the Panther to {a.robot} "
               f"(mount {name})\nrelease at t={ref.release_t:.3f}s here (as the Panther reference)\n"
               f"IK error: disc centre mean {e[:, 0].mean() * 100:.2f} cm max {e[:, 0].max() * 100:.2f} cm, "
               f"normal mean {math.degrees(e[:, 1].mean()):.2f} deg; peak joint speed / limit "
               f"{(speed / vmax).max():.2f}")
        np.savetxt(out_dir / f"{ref.name}.csv", np.c_[ref.t, q], delimiter=",", fmt="%.6f", header=hdr,
                   comments="# ")
        summary["references"][ref.name] = {
            "disc_err_cm_mean": float(e[:, 0].mean() * 100), "disc_err_cm_max": float(e[:, 0].max() * 100),
            "normal_err_deg_mean": float(math.degrees(e[:, 1].mean())),
            "normal_err_deg_at_release": float(math.degrees(e[k, 1])),
            "disc_err_cm_at_release": float(e[k, 0] * 100),
            "peak_speed_over_limit": float((speed / vmax).max())}
        print(ref.name, json.dumps(summary["references"][ref.name]))
    (out_dir / "mount.json").write_text(json.dumps(summary, indent=1))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
