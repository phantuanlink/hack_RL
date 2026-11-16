import sys, json, numpy as np, mujoco
from scipy.signal import savgol_filter
m = mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml"); d = mujoco.MjData(m)
qa = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]; va = [m.joint(f"joint_ra{i}").dofadr[0] for i in range(8)]
lim = np.array([31.0] * 5 + [11.0] * 3)
gb = m.body("gripper").id
for path in sys.argv[1:]:
    rows = np.loadtxt(path, delimiter=",", comments="#"); meta = json.load(open(path[:-4] + ".json"))
    t, q = rows[:, 0], rows[:, 1:]; k_rel = meta["gripper_open_index"] + 9   # frisbee leaves ~0.18 s after the command
    dt = t[1] - t[0]
    qd = savgol_filter(q, 7, 3, deriv=1, delta=dt, axis=0); qdd = savgol_filter(q, 9, 3, deriv=2, delta=dt, axis=0)
    tau = []
    for i, (a, b, c) in enumerate(zip(q, qd, qdd)):
        m.body_mass[gb] = m.body_mass[gb]  # frisbee mass ignored (0.175 kg, small vs arm)
        d.qpos[qa], d.qvel[va], d.qacc[va] = a, b, c
        mujoco.mj_inverse(m, d)
        tau.append(d.qfrc_inverse[va] + m.dof_frictionloss[va] * np.tanh(b / 0.02))
    tau = np.abs(np.array(tau)) / lim
    print(f"{path.split(chr(92))[-1].split('/')[-1]}: peak torque / limit per joint {tau.max(0).round(2)}; peak accel {np.abs(qdd).max(0).round(1)}")
