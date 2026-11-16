"""Franka Emika Panda (7 DOF + 2-finger hand) as a second, off-the-shelf arm for the same throw.

Model: MuJoCo Menagerie `franka_emika_panda/panda.xml` (Apache-2.0, assets/franka/LICENSE), with
measured link inertias. Its own actuators are replaced by mjlab position actuators with Menagerie's
gains plus velocity feed-forward (see KP), and the limits are Franka's datasheet values (joint speed 2.175 / 2.61 rad/s, acceleration
15 / 7.5 / 10 / 12.5 / 15 / 20 / 20 rad/s^2, torque 87 / 12 N m, grip force 70 N continuous).

The arm is mounted with its shoulder (joint 2) where the Panther's shoulder is, at the orientation that
tracks the human throw best (`python -m hack_rl.retarget_arm --robot franka` writes mount.json and the
retargeted reference throws). Only the arm: no table or mounting plate is modelled.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPdActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from hack_rl.reference import DATA_DIR
from hack_rl.robot.profile import ArmProfile, register

HERE = Path(__file__).resolve().parent
PANDA_XML = HERE / "assets" / "franka" / "panda.xml"
MOUNT_JSON = DATA_DIR / "franka" / "mount.json"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGERS = ("finger_joint1", "finger_joint2")
SHOULDER_IN_BASE = (0.0, 0.0, 0.333)                 # joint 2 axis in the link0 frame
VEL_LIMITS = (2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61)
ACC_LIMITS = (15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0)
# Menagerie's gains, with velocity feed-forward (BuiltinPdActuatorCfg) so kd damps the tracking error,
# not the motion; without it kd 450 x 2 rad/s asked for 10x the 87 N m limit and the wrist chattered.
KP = (4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0)
KD = (450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0)
EFFORT = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

DISC_RADIUS = 0.1375
PINCH = (0.0, 0.0, 0.098)                             # between the finger pads, hand frame (TCP 0.1034)
GRIP_REL_POS = (0.0, 0.0, PINCH[2] + DISC_RADIUS - 0.02)   # rim 2 cm between the pads
GRIP_REL_QUAT = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)  # disc normal = hand +y

# collisions kept: the wrist/hand against the base and shoulder (a real self-collision on a throw)
_COLLIDE = ("link0", "link1", "link2", "link6", "link7", "hand")
_EXCLUDE = (("link6", "hand"), ("link5", "link7"))


def _spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for a in list(spec.actuators):
        spec.delete(a)
    for k in list(spec.keys):
        spec.delete(k)
    for t in list(spec.tendons):
        spec.delete(t)
    # the finger coupling equality (two 15 g bodies, solref = the timestep) chattered: drive both fingers
    for e in list(spec.equalities):
        spec.delete(e)
    for b in spec.bodies:
        for g in b.geoms:
            if g.contype or g.conaffinity:
                keep = b.name in _COLLIDE
                g.contype, g.conaffinity = (1, 1) if keep else (0, 0)
    for b1, b2 in _EXCLUDE:
        spec.add_exclude(bodyname1=b1, bodyname2=b2)
    return spec


def load_mount() -> tuple[tuple, tuple]:
    if MOUNT_JSON.exists():
        m = json.loads(MOUNT_JSON.read_text())
        return tuple(m["base_pos"]), tuple(m["base_quat"])
    return (0.0, -0.2, 1.0), (1.0, 0.0, 0.0, 0.0)


def ik_model(base_pos, base_quat) -> mujoco.MjModel:
    """Plain MuJoCo model with the arm at the given mount (for retargeting)."""
    spec = _spec()
    link0 = spec.body("link0")
    link0.pos, link0.quat = list(base_pos), list(base_quat)
    return spec.compile()


def actuators() -> tuple:
    # joint impedance control on (position, velocity) of the commanded trajectory, like Franka's controller
    acts = [BuiltinPdActuatorCfg(target_names_expr=(j,), stiffness=kp, damping=kd, effort_limit=e, armature=0.1)
            for j, kp, kd, e in zip(ARM_JOINTS, KP, KD, EFFORT)]
    # fingers: 15 g each, so soft gains plus the gripper drive's reflected inertia (stiffer gains are
    # unstable at the 5 ms step and shake the hand, which showed up as a chattering grip force)
    acts.append(BuiltinPositionActuatorCfg(target_names_expr=FINGERS, stiffness=200.0, damping=4.0,
                                           effort_limit=70.0, armature=0.05))
    return tuple(acts)


def robot_cfg() -> EntityCfg:
    pos, quat = load_mount()
    return EntityCfg(
        spec_fn=_spec,
        init_state=EntityCfg.InitialStateCfg(pos=pos, rot=quat, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        articulation=EntityArticulationInfoCfg(actuators=actuators(), soft_joint_pos_limit_factor=0.95),
    )


@register("franka")
def franka_profile() -> ArmProfile:
    return ArmProfile(
        name="franka", robot_cfg=robot_cfg, arm_joints=ARM_JOINTS,
        finger_joints=FINGERS, finger_actuated=FINGERS, finger_closed=0.004, finger_open=0.04,
        gripper_body="hand", grip_rel_pos=GRIP_REL_POS, grip_rel_quat=GRIP_REL_QUAT, pinch_point=PINCH,
        vel_limits=VEL_LIMITS, accel_limits=ACC_LIMITS,
        gripper_open_time_s=0.10,
        # Franka hand: 70 N continuous grip, small fingertip pads (~1.8 x 1.8 cm)
        grip={"mu": 0.6, "squeeze_force_n": 70.0, "pad_half_tangential_m": 0.009, "pad_half_radial_m": 0.009},
        residual_scale=(0.3, 0.3, 0.3, 0.3, 0.6, 0.6, 0.6),
        reference_files=("franka/joints_trajectory_shoulder_rel_2.csv",
                         "franka/joints_trajectory_shoulder_rel_3.csv"),
        inertia_bodies=(r"link[1-7]", "hand"), arm_actuators=(r"joint[1-7]_pd_.*",),
        contact_primary="link6", contact_secondary=("link0", "link1", "link2"),
        viewer_body="link0", experiment="franka_frisbee_throw",
        joint_header="joint1..joint7",
    )
