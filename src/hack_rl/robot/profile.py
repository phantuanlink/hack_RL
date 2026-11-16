"""What the throw task needs to know about an arm. One profile per robot (Panther, Franka Panda)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from mjlab.entity import EntityCfg


@dataclass(frozen=True)
class ArmProfile:
    name: str
    robot_cfg: Callable[[], EntityCfg]
    arm_joints: tuple[str, ...]
    finger_joints: tuple[str, ...]          # every finger joint (all written to "closed" at reset)
    finger_actuated: tuple[str, ...]        # the ones with a position actuator (others follow by equality)
    finger_closed: float                    # finger joint position holding the disc
    finger_open: float
    gripper_body: str
    # pose of the frisbee in the gripper frame (the weld): disc normal = finger closing axis
    grip_rel_pos: tuple
    grip_rel_quat: tuple
    pinch_point: tuple                      # centre of the finger pads, gripper frame
    vel_limits: tuple                       # rad/s per arm joint (commanded targets never exceed these)
    accel_limits: tuple                     # rad/s^2 per arm joint
    gripper_open_time_s: float              # command -> the fingers let go
    # friction grip, nominal values (randomised around them): pad-on-disc friction, squeeze force per
    # finger, half sizes of the contact patch along the rim (tangential) and into the disc (radial)
    grip: dict
    residual_scale: tuple                   # rad per unit action, per arm joint
    reference_files: tuple
    inertia_bodies: tuple                   # regexes of the arm bodies whose inertia is randomised
    arm_actuators: tuple                    # regexes of the arm actuators (gain randomisation)
    contact_primary: str | None = None      # subtree that must not touch ...
    contact_secondary: tuple = ()           # ... these bodies
    viewer_body: str = ""
    experiment: str = ""
    joint_header: str = ""                  # column names for exported CSVs
    extras: dict = field(default_factory=dict)


_REGISTRY: dict[str, Callable[[], ArmProfile]] = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


def get_profile(name: str) -> ArmProfile:
    from hack_rl.robot import franka, panther  # noqa: F401  (importing registers them)
    return _REGISTRY[name]()
