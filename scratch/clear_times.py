import sys, numpy as np, mujoco
m = mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml"); d = mujoco.MjData(m)
rows = np.loadtxt(sys.argv[1], delimiter=",", comments="#")
adr = [m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)]
body = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "torso"]
arm = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("link_ra4", "link_ra5", "link_ra6", "link_ra7", "gripper", "finger_l", "finger_r")]
out = []
for r in rows:
    d.qpos[adr] = r[1:9]; mujoco.mj_forward(m, d)
    best = min((mujoco.mj_geomDistance(m, d, a, b, 0.3, None), m.geom(a).name, m.geom(b).name) for a in body for b in arm)
    out.append((r[0], *best))
close = [(t, dd, a, b) for t, dd, a, b in out if dd < 0.02]
print("within 2 cm:", [(round(t, 2), round(dd * 100, 1), a, b) for t, dd, a, b in close[::2]])
