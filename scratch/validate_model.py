import numpy as np, mujoco
m = mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml"); d = mujoco.MjData(m)
qa = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]; va = [m.joint(f"joint_ra{i}").dofadr[0] for i in range(8)]
import json; K = np.array([json.load(open("scratch/identified.json"))["joints"][f"joint_ra{i}"]["nm_per_effort_unit"] for i in range(8)]); K[6] = K[5]
for n in ("run3", "fast1", "fast2"):
    z = np.load(f"scratch/prep_{n}.npz")
    tau = []
    for q, qd, qdd in zip(z["q"], z["qd"], z["qdd"]):
        d.qpos[:] = 0; d.qvel[:] = 0; d.qacc[:] = 0
        d.qpos[qa], d.qvel[va], d.qacc[va] = q, qd, qdd
        mujoco.mj_inverse(m, d)
        tau.append(d.qfrc_inverse[va] + m.dof_frictionloss[va] * np.tanh(qd / 0.02))
    tau = np.array(tau); meas = z["e"] * K
    r2 = 1 - ((meas - tau) ** 2).sum(0) / ((meas - meas.mean(0)) ** 2).sum(0)
    print(f"{n}: measured torque std {meas.std(0).round(2)} N m")
    print(f"{n}: peak measured |tau| {np.abs(meas).max(0).round(1)} N m, model {np.abs(tau).max(0).round(1)}")
    print(f"{n}: MuJoCo model R2 {r2.round(2)}  rms err {np.sqrt(((meas - tau) ** 2).mean(0)).round(2)} N m")
