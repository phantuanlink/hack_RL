"""Build the MuJoCo model of the Panther's right arm (+ torso, + gripper) from the vendor URDF.

    uv run python -m hack_rl.robot.build_mjcf            # writes robot/assets/panther_right_arm.xml

Kinematics (joint origins, axes, limits) come from the URDF. Masses, centres of mass and inertias come
from `sysid_params.yaml`: today the vendor's CAD values ("assume system ID is done"); tomorrow replace
that file with the identified parameters and re-run. Actuator gains, joint damping/armature/friction
and velocity/effort limits also live there.

The URDF has no usable collision geometry (STL meshes live on the robot), so links get capsules between
consecutive joint origins, the torso a box, the parked left arm a capsule, and the gripper boxes. The
real gripper (unix_gripper3, "parallel" mode, command 0.0 open .. 1.0 closed) is modelled as two
fingers on slide joints driven together by one position actuator.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ARM = [f"joint_ra{i}" for i in range(8)]
LINKS = [f"link_ra{i}" for i in range(8)]
GRIPPER_LINKS = ["right_connector", "link_rf", "link_rf00", "link_rf01", "link_rf02", "link_rf03",
                 "link_rf10", "link_rf11", "link_rf12", "link_rf13"]


def _v(s, n=3):
    return np.array([float(x) for x in (s or " ".join(["0"] * n)).split()])


def _fmt(a):
    return " ".join(f"{x:.6g}" for x in np.asarray(a).ravel())


def read_urdf(path):
    r = ET.parse(path).getroot()
    joints = {j.get("name"): j for j in r.findall("joint")}
    links = {l.get("name"): l for l in r.findall("link")}
    return joints, links


def joint_info(j):
    o = j.find("origin")
    a = j.find("axis")
    lim = j.find("limit")
    return {
        "xyz": _v(o.get("xyz") if o is not None else None),
        "rpy": _v(o.get("rpy") if o is not None else None),
        "axis": _v(a.get("xyz")) if a is not None else np.array([0, 0, 1.0]),
        "lower": float(lim.get("lower")) if lim is not None else 0.0,
        "upper": float(lim.get("upper")) if lim is not None else 0.0,
        "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
    }


def link_inertial(link):
    i = link.find("inertial")
    if i is None:
        return None
    I = i.find("inertia").attrib
    return {"mass": float(i.find("mass").get("value")),
            "com": [float(c) for c in _v(i.find("origin").get("xyz"))],
            "inertia": [float(I[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]}


def _rpy_quat(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def lump(inertials_with_offsets):
    """Combine rigid bodies (mass, com, full inertia at com) given offsets into one inertial."""
    m = sum(x["mass"] for x, _ in inertials_with_offsets)
    com = sum(x["mass"] * (np.array(x["com"]) + off) for x, off in inertials_with_offsets) / m
    I = np.zeros((3, 3))
    for x, off in inertials_with_offsets:
        ixx, iyy, izz, ixy, ixz, iyz = x["inertia"]
        Ic = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        d = np.array(x["com"]) + off - com
        I += Ic + x["mass"] * (d @ d * np.eye(3) - np.outer(d, d))
    return {"mass": float(m), "com": [float(c) for c in com],
            "inertia": [float(v) for v in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])]}


def write_default_params(urdf, out):
    """sysid_params.yaml from the URDF's CAD inertials (the 'pretend system ID is done' values)."""
    joints, links = read_urdf(urdf)
    params = {
        "source": f"CAD inertials from {Path(urdf).name} (placeholder until real system ID)",
        "links": {n: link_inertial(links[n]) for n in ["body_link"] + LINKS},
    }
    # gripper: everything rigidly attached below link_ra7, lumped into one body in link_ra7's frame
    off = {"link_ra7": np.zeros(3)}
    parts = []
    for name in GRIPPER_LINKS:
        j = next(jj for jj in joints.values() if jj.find("child").get("link") == name)
        ji = joint_info(j)
        off[name] = off[ji["parent"]] + ji["xyz"]
        li = link_inertial(links[name])
        if li:
            parts.append((li, off[name]))
    params["links"]["gripper"] = lump(parts)
    params["joints"] = {n: {"damping": 0.1, "armature": 0.005, "frictionloss": 0.02,
                            "kp": 30.0, "kd": 1.5, "effort_limit": 10.0, "velocity_limit": 3.0, "accel_limit": 25.0} for n in ARM}
    params["gripper"] = {"stroke_m": 0.03, "kp": 200.0, "kd": 5.0, "force_limit": 40.0,
                         "open_time_s": 0.08, "note": "real opening speed unmeasured"}
    Path(out).write_text(yaml.safe_dump(params, sort_keys=False))
    return params


def build(urdf=ASSETS / "panther_full.urdf", params_path=HERE / "sysid_params.yaml",
          out=ASSETS / "panther_right_arm.xml"):
    joints, _ = read_urdf(urdf)
    if not Path(params_path).exists():
        write_default_params(urdf, params_path)
    P = yaml.safe_load(Path(params_path).read_text())
    L, J = P["links"], P["joints"]

    m = ET.Element("mujoco", model="panther_right_arm")
    ET.SubElement(m, "compiler", angle="radian", autolimits="true")
    default = ET.SubElement(m, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="0.8 0.02 0.001",
                  rgba="0.75 0.77 0.8 1")
    wb = ET.SubElement(m, "worldbody")

    def inertial(body, li):
        ixx, iyy, izz, ixy, ixz, iyz = li["inertia"]
        ET.SubElement(body, "inertial", pos=_fmt(li["com"]), mass=f"{li['mass']:.6g}",
                      fullinertia=_fmt([ixx, iyy, izz, ixy, ixz, iyz]))

    # torso = body_link, fixed; its origin is at shoulder height. mjlab wraps fixed-base entities.
    torso = ET.SubElement(wb, "body", name="torso", pos="0 0 0")
    inertial(torso, L["body_link"])
    ET.SubElement(torso, "geom", name="torso_box", type="box", size="0.10 0.12 0.28", pos="-0.02 0 -0.26",
                  rgba="0.35 0.38 0.45 1")
    ET.SubElement(torso, "geom", name="head", type="sphere", size="0.10", pos="0 0 0.25", rgba="0.35 0.38 0.45 1")
    ET.SubElement(torso, "geom", name="left_arm_parked", type="capsule", fromto="0 0.24 0 0 0.27 -0.55",
                  size="0.04", rgba="0.55 0.57 0.6 1")
    ET.SubElement(torso, "site", name="torso_site", pos="0 0 0")

    radii = [0.045, 0.045, 0.04, 0.04, 0.035, 0.035, 0.03, 0.03]
    parent = torso
    for k, (jn, ln) in enumerate(zip(ARM, LINKS)):
        ji = joint_info(joints[jn])
        q = _rpy_quat(ji["rpy"])
        b = ET.SubElement(parent, "body", name=ln, pos=_fmt(ji["xyz"]), quat=_fmt(q))
        inertial(b, L[ln])
        jp = J[jn]
        ET.SubElement(b, "joint", name=jn, type="hinge", axis=_fmt(ji["axis"]),
                      range=f"{ji['lower']:.6g} {ji['upper']:.6g}", damping=f"{jp['damping']}",
                      armature=f"{jp['armature']}", frictionloss=f"{jp['frictionloss']}",
                      actuatorfrcrange=f"{-jp['effort_limit']} {jp['effort_limit']}")
        # capsule from this joint to the next joint origin (in this link's frame)
        if k + 1 < len(ARM):
            nxt = joint_info(joints[ARM[k + 1]])["xyz"]
            if np.linalg.norm(nxt) > 0.02:
                ET.SubElement(b, "geom", name=f"{ln}_col", type="capsule", fromto=_fmt(np.r_[np.zeros(3), nxt]),
                              size=f"{radii[k]}")
        parent = b

    # gripper (rigid lump in link_ra7 frame) + two fingers on slides along y
    ra7 = parent
    g = ET.SubElement(ra7, "body", name="gripper", pos="0 0 0")
    inertial(g, L["gripper"])
    rf = np.array([0.0054849, -0.0029567, -0.083332]) + np.array([0, 0, -0.05])   # link_rf in link_ra7 frame
    ET.SubElement(g, "geom", name="gripper_mount", type="cylinder", fromto=_fmt(np.r_[0, 0, 0, 0, 0, -0.05]),
                  size="0.03", rgba="0.3 0.3 0.32 1")
    ET.SubElement(g, "geom", name="gripper_palm", type="box", size="0.03 0.07 0.04", pos=_fmt(rf + [0, 0, 0.01]),
                  rgba="0.3 0.3 0.32 1")
    stroke = P["gripper"]["stroke_m"]
    for side, s in (("l", 1), ("r", -1)):
        f = ET.SubElement(g, "body", name=f"finger_{side}", pos=_fmt(rf + [0, s * 0.059, -0.035]))
        ET.SubElement(f, "inertial", pos="0 0 -0.02", mass="0.05", diaginertia="1e-5 1e-5 1e-5")
        # q = 0 open, q = stroke closed (moves toward the centre)
        ET.SubElement(f, "joint", name=f"finger_{side}_joint", type="slide", axis=_fmt([0, -s, 0]),
                      range=f"0 {stroke}", damping="1")
        ET.SubElement(f, "geom", name=f"finger_{side}_geom", type="box", size="0.012 0.006 0.035",
                      pos="0 0 -0.02", rgba="0.2 0.2 0.22 1")
    # where the frisbee rim is pinched: between the finger tips
    ET.SubElement(g, "site", name="grip_site", pos=_fmt(rf + [0, 0, -0.065]), size="0.01", rgba="1 0.5 0 1")
    ET.SubElement(m, "equality").append(ET.Element("joint", name="finger_mimic", joint1="finger_r_joint",
                                                   joint2="finger_l_joint", polycoef="0 1 0 0 0"))
    ET.SubElement(m, "contact").extend([
        ET.Element("exclude", body1="torso", body2="link_ra0"),
        ET.Element("exclude", body1="torso", body2="link_ra1"),
        ET.Element("exclude", body1="link_ra7", body2="gripper"),
        ET.Element("exclude", body1="link_ra5", body2="gripper"),
        ET.Element("exclude", body1="link_ra6", body2="gripper"),
        ET.Element("exclude", body1="link_ra5", body2="link_ra7"),
        ET.Element("exclude", body1="link_ra4", body2="link_ra6"),
        ET.Element("exclude", body1="link_ra5", body2="finger_l"),
        ET.Element("exclude", body1="link_ra5", body2="finger_r"),
        ET.Element("exclude", body1="gripper", body2="finger_l"),
        ET.Element("exclude", body1="gripper", body2="finger_r"),
        ET.Element("exclude", body1="finger_l", body2="finger_r"),
    ])
    ET.indent(m)
    Path(out).write_text(ET.tostring(m, encoding="unicode"))
    return out


FRISBEE_XML = """<mujoco model="frisbee">
  <!-- 175 g ultimate disc, 27.5 cm diameter. Inertia from Hummel (2003): Ixx=Iyy=1.219e-3, Izz=2.352e-3.
       Its own z axis is the disc normal (the spin axis). No collisions: while held it moves with the gripper,
       and the episode ends shortly after release (only the release state is scored). -->
  <worldbody>
    <body name="frisbee">
      <freejoint name="frisbee_free"/>
      <inertial pos="0 0 0" mass="0.175" diaginertia="0.001219 0.001219 0.002352"/>
      <geom name="disc" type="cylinder" size="0.1375 0.008" rgba="0.95 0.55 0.1 1" contype="0" conaffinity="0"/>
      <geom name="rim" type="cylinder" size="0.1375 0.016" pos="0 0 -0.008" rgba="0.95 0.55 0.1 0.35"
            contype="0" conaffinity="0"/>
      <site name="frisbee_center" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_frisbee(out=ASSETS / "frisbee.xml"):
    Path(out).write_text(FRISBEE_XML)
    return out


if __name__ == "__main__":
    print("wrote", build())
    print("wrote", write_frisbee())
