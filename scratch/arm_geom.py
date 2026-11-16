import numpy as np, mujoco
from hack_rl.robot import panther as P
m = mujoco.MjModel.from_xml_path(str(P.ROBOT_XML)); d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for i in range(8):
    j = m.joint(f"joint_ra{i}")
    print(f"joint_ra{i} anchor {d.xanchor[j.id].round(3)} axis {d.xaxis[j.id].round(2)} range {m.jnt_range[j.id].round(2)}")
gb = m.body("gripper").id
print("grip_site", d.site_xpos[m.site("grip_site").id].round(3))
