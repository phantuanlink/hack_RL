import json, numpy as np, mujoco, math
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML)); d = mujoco.MjData(m)
print("ra0 range (rad):", m.jnt_range[m.joint("joint_ra0").id].round(3), " ra7 range:", m.jnt_range[m.joint("joint_ra7").id].round(3))
adr = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]
gb = m.body(P.GRIPPER_BODY).id
for r in (2, 3):
    base = f"exports/2026-11-15_11-17-42_dr2a/model_1499/throw_joints_trajectory_shoulder_rel_{r}"
    rows = np.loadtxt(base + ".csv", delimiter=",", comments="#"); meta = json.load(open(base + ".json"))
    t, q = rows[:, 0], rows[:, 1:]; k = meta["gripper_open_index"]
    v = np.gradient(q, t, axis=0)
    print(f"ref{r}: ra0 from {q[:k+10,0].min():.2f} to {q[:k+10,0].max():.2f} rad; at release ra0 {q[k,0]:.2f}, ra0 speed {v[k,0]:+.2f}, ra7 speed {v[k,7]:+.2f} rad/s; "
          f"joints at the speed cap near release: {[i for i in range(8) if abs(v[k-5:k+5, i]).max() > 1.9]}")
    vel = meta["velocity_mps"]; print(f"   release velocity {np.round(vel,2)} -> heading {math.degrees(math.atan2(vel[1], vel[0])):+.1f} deg; open command at t={meta['gripper_open_command_t']:.2f}, frisbee free at {meta['frisbee_leaves_t']:.2f}")
