import numpy as np, mujoco
from hack_rl import enlarge_references as E
from hack_rl.reference import load_reference, DATA_DIR, DEFAULT_FILES
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML)); d = mujoco.MjData(m)
adr = np.array([m.joint(f"joint_ra{i}").qposadr[0] for i in range(8)])
arm_ids = {m.body(b).id for b in E.ARM_BODIES}
arm = [g for g in range(m.ngeom) if m.geom_bodyid[g] in arm_ids]
for f in DEFAULT_FILES:
    ref = load_reference(DATA_DIR / f); worst = (1e9,)
    for i, q in enumerate(ref.q[::2]):
        d.qpos[adr] = q; mujoco.mj_forward(m, d)
        for gname in E.BODY_GEOMS:
            for ga in arm:
                gap = E._gap(m, d, m.geom(gname).id, ga)
                if gap < worst[0]: worst = (gap, ref.t[2 * i], gname, m.geom(ga).name or m.body(m.geom_bodyid[ga]).name)
    print(f, "closest:", round(worst[0] * 100, 1), "cm at t", round(worst[1], 2), worst[2], worst[3])
