import numpy as np, mujoco, math
from hack_rl.reference import load_reference, DATA_DIR
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML)); d = mujoco.MjData(m)
adr = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]
mujoco.mj_forward(m, d)
for i in range(3):
    j = m.joint(f"joint_ra{i}"); b = m.jnt_bodyid[j.id]
    print(f"joint_ra{i} axis (world, at q=0): {(d.xmat[b].reshape(3,3) @ m.jnt_axis[j.id]).round(3)}")
gb = m.body(P.GRIPPER_BODY).id
for f in ("adapted/joints_trajectory_shoulder_rel_2.csv", "adapted/joints_trajectory_shoulder_rel_3.csv"):
    ref = load_reference(DATA_DIR / f); dt = ref.t[1] - ref.t[0]
    pos = []
    for q in ref.q:
        d.qpos[adr] = q; mujoco.mj_kinematics(m, d)
        pos.append(d.xpos[gb] + d.xmat[gb].reshape(3, 3) @ np.array(P.GRIP_REL_POS))
    pos = np.array(pos); v = np.gradient(pos, dt, axis=0); k = int(round(ref.release_t / dt))
    yaw = math.degrees(math.atan2(v[k, 1], v[k, 0]))
    print(f"{ref.name}: disc velocity at release {v[k].round(2)} m/s -> heading {yaw:+.1f} deg from forward (+x), speed {np.linalg.norm(v[k]):.2f}")
