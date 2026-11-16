import mujoco
m=mujoco.MjModel.from_xml_path("src/hack_rl/robot/assets/panther_right_arm.xml")
for n in ["torso_box","head","left_arm_parked","link_ra3_col","link_ra4_col","link_ra5_col","gripper_mount","gripper_palm","finger_l_geom","finger_r_geom"]:
    g=m.geom(n); print(n, "contype", m.geom_contype[g.id], "conaff", m.geom_conaffinity[g.id], "body", m.body(m.geom_bodyid[g.id]).name)
print([ (m.body(m.exclude_signature[i]>>16).name, m.body(m.exclude_signature[i]&0xFFFF).name) for i in range(m.nexclude)])
