import sys, json, numpy as np, mujoco
from scipy.signal import savgol_filter
from hack_rl import enlarge_references as E
m = mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml"); d = mujoco.MjData(m)
qa = np.array([m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]); va = [m.joint(f"joint_ra{i}").dofadr[0] for i in range(8)]
lim = np.array([31.0] * 5 + [11.0] * 3)
arm_ids = {m.body(b).id for b in E.ARM_BODIES}
for path in sys.argv[1:]:
    rows = np.loadtxt(path, delimiter=",", comments="#"); t, q = rows[:, 0], rows[:, 1:]; dt = t[1] - t[0]
    qd = savgol_filter(q, 5, 2, deriv=1, delta=dt, axis=0); qdd = savgol_filter(q, 7, 2, deriv=2, delta=dt, axis=0)
    tau = []
    for a, b, c in zip(q, qd, qdd):
        d.qpos[qa], d.qvel[va], d.qacc[va] = a, b, c; mujoco.mj_inverse(m, d)
        tau.append(np.abs(d.qfrc_inverse[va] + m.dof_frictionloss[va] * np.tanh(b / 0.02)))
    tau = np.array(tau) / lim
    gap = E.min_gap(m, d, q[::2], qa, arm_ids)
    tag = path.replace("\\", "/").split("exports/")[-1]
    print(f"{tag}\n   peak speed per joint (rad/s) {np.abs(qd).max(0).round(1)}\n   peak accel {np.abs(qdd).max(0).round(0)}\n   peak torque / limit {tau.max(0).round(2)}   min gap to torso/head {gap*100:.1f} cm")
