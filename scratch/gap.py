import numpy as np, mujoco
m = mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml"); d = mujoco.MjData(m)
rows = np.loadtxt("exports/test0_dr2a_half_speed/test0_ref2_half_speed.csv", delimiter=",", comments="#")
adr = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]
tb = m.geom("torso_box").id
def gap(gname):
    g = m.geom(gname).id
    # sample the geom's box corners / capsule points, express in torso_box frame, distance to the box
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
        pts = [c + R @ (np.array(sg) * s) for sg in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T]
    else:
        pts = [c]
    Rt = d.geom_xmat[tb].reshape(3, 3); ct = d.geom_xpos[tb]; st = m.geom_size[tb]
    best = 1e9
    for p in pts:
        lp = Rt.T @ (p - ct)
        out = np.maximum(np.abs(lp) - st, 0)
        dist = np.linalg.norm(out) if out.any() else -np.min(st - np.abs(lp))
        best = min(best, dist)
    return best
for t_sel in (0.0, 0.37, 1.32):
    i = int(np.argmin(np.abs(rows[:, 0] - t_sel)))
    d.qpos[adr] = rows[i, 1:9]; mujoco.mj_forward(m, d)
    print(f"t={rows[i,0]:.2f}s  palm corner-to-torso-box gap {gap('gripper_palm')*100:+.1f} cm, finger_l {gap('finger_l_geom')*100:+.1f} cm, "
          f"geomDistance {mujoco.mj_geomDistance(m, d, tb, m.geom('gripper_palm').id, 0.3, None)*100:.2f} cm")
