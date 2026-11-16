"""mjlab entity configs: Panther right arm (+ torso, gripper) and the frisbee, plus the grip weld.

Everything robot-specific that system identification will change (masses, inertias, PD gains, effort
and velocity limits, gripper timing) is read from `sysid_params.yaml`. Rebuild the MJCF after editing it:
    uv run python -m hack_rl.robot.build_mjcf
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mjlab.actuator import BuiltinPdActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from hack_rl.robot.profile import ArmProfile, register

HERE = Path(__file__).resolve().parent
ROBOT_XML = HERE / "assets" / "panther_right_arm.xml"
FRISBEE_XML = HERE / "assets" / "frisbee.xml"
PARAMS_YAML = HERE / "sysid_params.yaml"

ARM_JOINTS = tuple(f"joint_ra{i}" for i in range(8))
GRIPPER_JOINT = "finger_l_joint"          # finger_r follows through an equality constraint
GRIPPER_BODY = "gripper"
TORSO_HEIGHT = 1.3632                     # body_link above the floor with the lift at 0 (URDF body_joint)

# Frisbee pose in the gripper frame (the weld's relative pose). The rim is pinched between the fingers,
# so the disc normal (frisbee z) is the fingers' closing axis (gripper y) and the disc centre sits one
# radius beyond the finger tips (gripper -z).
DISC_RADIUS = 0.1375
LINK_RF_IN_RA7 = np.array([0.0054849, -0.0029567, -0.133332])     # right_connector + link_rf offsets
GRIP_REL_POS = tuple((LINK_RF_IN_RA7 + [0.0, 0.0, -0.065 - DISC_RADIUS + 0.02]).tolist())
GRIP_REL_QUAT = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)   # -90 deg about x
WELD_NAME = "grip_weld"
WELD_TIMECONST = 0.01                     # s; 0.005 (= the timestep) made the grip force flip sign every step


@lru_cache
def load_params(path: str = str(PARAMS_YAML)) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _robot_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(ROBOT_XML))


def _frisbee_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(FRISBEE_XML))


def arm_actuators(params: dict | None = None) -> tuple:
    """Joint servos: PD on (commanded position, commanded velocity), torque-limited. Velocity feed-forward
    like a servo drive following a trajectory (a plain position actuator would damp toward zero speed)."""
    p = params or load_params()
    acts = [
        BuiltinPdActuatorCfg(
            target_names_expr=(name,),
            stiffness=float(p["joints"][name]["kp"]),
            damping=float(p["joints"][name]["kd"]),
            effort_limit=float(p["joints"][name]["effort_limit"]),
        )
        for name in ARM_JOINTS
    ]
    g = p["gripper"]
    acts.append(BuiltinPositionActuatorCfg(target_names_expr=(GRIPPER_JOINT,), stiffness=float(g["kp"]),
                                           damping=float(g["kd"]), effort_limit=float(g["force_limit"])))
    return tuple(acts)


def joint_velocity_limits(params: dict | None = None) -> dict[str, float]:
    p = params or load_params()
    return {name: float(p["joints"][name]["velocity_limit"]) for name in ARM_JOINTS}


def torso_height(params: dict | None = None) -> float:
    """body_link height: the lift (body_joint, axis -z) lowers it from TORSO_HEIGHT by sysid lift_m."""
    p = params or load_params()
    return TORSO_HEIGHT - float(p.get("lift_m", 0.0))


def robot_cfg() -> EntityCfg:
    stroke = float(load_params()["gripper"]["stroke_m"])
    return EntityCfg(
        spec_fn=_robot_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, torso_height()),
            joint_pos={"finger_.*": stroke, ".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(actuators=arm_actuators(), soft_joint_pos_limit_factor=0.95),
    )


def frisbee_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=_frisbee_spec,
                     init_state=EntityCfg.InitialStateCfg(pos=(0.4, -0.3, 0.8)))


def make_grip_weld(gripper_body: str, rel_pos, rel_quat):
    """Scene-level hook: weld the frisbee to the gripper (switched off per env at release)."""
    def add(spec: mujoco.MjSpec) -> None:
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=WELD_NAME, name1=f"robot/{gripper_body}",
                               name2="frisbee/frisbee", objtype=mujoco.mjtObj.mjOBJ_BODY)
        data = np.zeros(len(eq.data))           # assign the whole array: slice writes don't reach the spec
        data[3:6] = rel_pos
        data[6:10] = rel_quat
        data[10] = 1.0                           # torque scale
        eq.data = data
        eq.solref = [WELD_TIMECONST, 1.0]        # stiff, but >= 2 physics steps or the weld chatters
    return add


add_grip_weld = make_grip_weld(GRIPPER_BODY, GRIP_REL_POS, GRIP_REL_QUAT)


@register("panther")
def panther_profile() -> ArmProfile:
    p = load_params()
    from hack_rl.reference import DEFAULT_FILES
    return ArmProfile(
        name="panther", robot_cfg=robot_cfg, arm_joints=ARM_JOINTS,
        finger_joints=("finger_l_joint", "finger_r_joint"), finger_actuated=(GRIPPER_JOINT,),
        finger_closed=float(p["gripper"]["stroke_m"]), finger_open=0.0, gripper_body=GRIPPER_BODY,
        grip_rel_pos=GRIP_REL_POS, grip_rel_quat=GRIP_REL_QUAT,
        pinch_point=tuple((LINK_RF_IN_RA7 + [0.0, 0.0, -0.065]).tolist()),
        vel_limits=tuple(joint_velocity_limits(p)[j] for j in ARM_JOINTS),
        accel_limits=tuple(float(p["joints"][j].get("accel_limit", 25.0)) for j in ARM_JOINTS),
        gripper_open_time_s=float(p["gripper"]["open_time_s"]),
        grip={k: float(v) for k, v in p["grip"].items() if isinstance(v, (int, float))},
        residual_scale=(0.3, 0.3, 0.3, 0.3, 0.5, 0.6, 0.6, 1.0),
        reference_files=DEFAULT_FILES,
        inertia_bodies=(r"link_ra.*", "gripper"), arm_actuators=(r"joint_ra.*_pd_.*",),
        contact_primary="link_ra3", contact_secondary=("torso",),   # torso box, head, parked left arm
        viewer_body="torso", experiment="panther_frisbee_throw",
        joint_header="joint_ra0..joint_ra7",
    )
