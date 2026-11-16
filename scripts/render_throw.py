"""Render an exported throw (CSV + JSON from hack_rl.export_rollout) to an MP4.

    uv run python scripts/render_throw.py exports/<run>/throw_<ref>.csv [--out video.mp4]

Kinematic playback (no physics): the arm follows the CSV's joint targets; the frisbee sits in the grip until
the release time in the JSON, then flies ballistically with the recorded release velocity and spin (no
aerodynamics, ~0.6 s shown). The disc's spin axis is drawn as a short line while it flies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np

from hack_rl.robot.panther import FRISBEE_XML, GRIP_REL_POS, GRIP_REL_QUAT, ROBOT_XML, torso_height


def build_model():
    s = mujoco.MjSpec()
    s.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[6, 6, 0.1], rgba=[0.75, 0.32, 0.28, 1])
    for k in range(1, 6):                                   # distance marks every metre in front
        s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 1.5, 0.001], pos=[k, 0, 0.001],
                             rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
    s.attach(mujoco.MjSpec.from_file(str(ROBOT_XML)), prefix="robot/",
             frame=s.worldbody.add_frame(pos=[0, 0, torso_height()]))
    s.attach(mujoco.MjSpec.from_file(str(FRISBEE_XML)), prefix="frisbee/", frame=s.worldbody.add_frame())
    s.visual.global_.offwidth, s.visual.global_.offheight = 1280, 720
    s.visual.headlight.ambient = [0.45, 0.45, 0.45]
    s.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    s.worldbody.add_light(pos=[2, -3, 4], dir=[-0.3, 0.5, -0.8], diffuse=[0.6, 0.6, 0.6])
    m = s.compile()
    return m, mujoco.MjData(m)


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fly-s", type=float, default=0.6)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--slowmo", type=float, default=1.0, help="play this many times slower (e.g. 4 for a 0.7 s whip)")
    a = ap.parse_args()
    rows = np.loadtxt(a.csv, delimiter=",", comments="#")
    t, Q = rows[:, 0], rows[:, 1:9]
    meta = json.loads(a.csv.with_suffix(".json").read_text())
    t_rel = meta.get("frisbee_leaves_t", meta.get("gripper_open_command_t", float("inf")))
    v_rel = np.array(meta.get("velocity_mps", [0, 0, 0]))
    w_rel = np.array(meta.get("angular_velocity_radps", [0, 0, 0]))

    m, d = build_model()
    arm_q = [m.jnt_qposadr[m.joint(f"robot/joint_ra{i}").id] for i in range(8)]
    fing = [m.jnt_qposadr[m.joint(f"robot/finger_{s}_joint").id] for s in "lr"]
    fq = m.jnt_qposadr[m.joint("frisbee/frisbee_free").id]
    g = m.body("robot/gripper").id
    stroke = 0.03

    out = a.out or a.csv.with_suffix(".mp4")
    renderer = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.7, -0.3, torso_height() - 0.25]   # robot's right side; the throw (+x) goes left -> right
    cam.distance, cam.azimuth, cam.elevation = 2.6, 90, -10
    writer = imageio_ffmpeg.write_frames(str(out), (1280, 720), fps=a.fps, codec="libx264", pix_fmt_out="yuv420p",
                                         quality=7)
    writer.send(None)
    T = np.arange(0, min(t[-1], t_rel + a.fly_s) + 1e-9, 1 / (a.fps * a.slowmo))
    p_rel = q_rel = None
    for tt in T:
        q = np.array([np.interp(tt, t, Q[:, j]) for j in range(8)])
        d.qpos[arm_q] = q
        held = tt < t_rel
        d.qpos[fing] = stroke if held else 0.0
        mujoco.mj_kinematics(m, d)
        if held:
            gq = np.zeros(4)
            mujoco.mju_mat2Quat(gq, d.xmat[g])
            p = d.xpos[g] + d.xmat[g].reshape(3, 3) @ np.array(GRIP_REL_POS)
            quat = quat_mul(gq, np.array(GRIP_REL_QUAT))
            p_rel, q_rel = p, quat
        else:
            dt = tt - t_rel
            p = p_rel + v_rel * dt + 0.5 * np.array([0, 0, -9.81]) * dt * dt
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, w_rel / (np.linalg.norm(w_rel) + 1e-9), np.linalg.norm(w_rel) * dt)
            quat = quat_mul(dq, q_rel)
        d.qpos[fq:fq + 3] = p
        d.qpos[fq + 3:fq + 7] = quat
        mujoco.mj_kinematics(m, d)
        renderer.update_scene(d, cam)
        frame = renderer.render()
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
