import json, numpy as np, mujoco
m = mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml"); d = mujoco.MjData(m)
qa = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]; va = [m.joint(f"joint_ra{i}").dofadr[0] for i in range(8)]
K = np.array([json.load(open("scratch/identified.json"))["joints"][f"joint_ra{i}"]["nm_per_effort_unit"] for i in range(8)]); K[6] = K[5]
X, Y = [], []
for n in ("fast1", "fast2", "run3"):
    z = np.load(f"scratch/prep_{n}.npz")
    for q, qd, qdd, e in zip(z["q"], z["qd"], z["qdd"], z["e"]):
        d.qpos[qa], d.qvel[va], d.qacc[va] = q, qd, qdd
        mujoco.mj_inverse(m, d)
        tau = d.qfrc_inverse[va[0]] + m.dof_frictionloss[va[0]] * np.tanh(qd[0] / 0.02)
        X.append([qdd[0], qd[0]]); Y.append(e[0] * K[0] - tau)
X, Y = np.array(X), np.array(Y)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
mujoco.mj_setConst(m, d)
print(f"extra ra0 armature (kg m^2) {coef[0]:.3f}, extra viscous {coef[1]:.3f}; model ra0 armature now {m.dof_armature[va[0]]:.3f}")
r_before = np.sqrt((Y ** 2).mean()); r_after = np.sqrt(((Y - X @ coef) ** 2).mean())
print(f"ra0 torque rms error {r_before:.2f} -> {r_after:.2f} N m")
