"""Frisbee throw with one arm (Panther right arm or Franka Panda): environment configuration."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import events as mjlab_events
from mjlab.envs.mdp import rewards as mjlab_rewards
from mjlab.envs.mdp import terminations as mjlab_terms
from mjlab.envs.mdp.observations import last_action
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import dataclasses

from hack_rl.robot.panther import frisbee_cfg, make_grip_weld
from hack_rl.robot.profile import get_profile
from hack_rl.tasks.throw import mdp

# joint friction (N m) to randomise over: the Panther joints are small (10 N m), the Franka's large
JOINT_FRICTION = {"panther": (0.0, 0.1), "franka": (0.0, 1.0)}
_DEG2 = 0.0349                                    # 2 deg in rad
POSTURE_RANGE = {"z": (-0.03, 0.03), "roll": (-_DEG2, _DEG2), "pitch": (-_DEG2, _DEG2), "yaw": (-_DEG2, _DEG2)}
# command latency of the position controller, in 5 ms physics steps (ROS 2 -> controller -> motor bus)
COMMAND_LAG_STEPS = (0, 6)


def throw_env_cfg(robot: str = "panther", play: bool = False, reference_ids: tuple | None = None,
                  open_loop: bool = True, vel_cap: float | None = None, accel_cap: float | None = None,
                  limit_reaction: bool = False, heading_range: tuple | None = None, heading_play: float = 0.0,
                  release_delay_range: tuple | None = None, release_latency_range: tuple = (0.0, 0.03),
                  command_lag_steps: tuple = COMMAND_LAG_STEPS, reference_files: tuple | None = None,
                  residual_scale: tuple | None = None, goal_speed: float | None = None,
                  imitate_joints: int | None = None, goal_spin: float | None = None,
                  goal_weights: dict | None = None, imitation: bool = True,
                  grip_curriculum: tuple | None = None, aggression_range: tuple | None = None,
                  aggression_play: float = 1.0, grip_strength: float = 1.0,
                  speed_range: tuple | None = None, release_shift_s: float | None = None,
                  joint_max_override: dict | None = None, pitch_range: tuple | None = None,
                  pitch_play: float | None = None) -> ManagerBasedRlEnvCfg:
    """vel_cap / accel_cap: commanded joint speed / acceleration caps (dr2a 2.0 / 10, dr2b 2.5 / 15).
    limit_reaction: penalise shoulder torque > 70 % of its limit and the arm's push on the torso (dr2b)."""
    prof = get_profile(robot)
    arm = SceneEntityCfg("robot", joint_names=prof.arm_joints)
    critic_terms = {
        "phase": ObservationTermCfg(func=mdp.throw_phase),
        "ref_preview": ObservationTermCfg(func=mdp.reference_preview, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_pos": ObservationTermCfg(func=mdp.arm_joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObservationTermCfg(func=mdp.arm_joint_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "frisbee": ObservationTermCfg(func=mdp.frisbee_motion, noise=Unoise(n_min=-0.02, n_max=0.02)),
        "actions": ObservationTermCfg(func=last_action),
    }
    if heading_range is not None:                       # goal-conditioned: both networks see the target heading
        critic_terms["heading"] = ObservationTermCfg(func=mdp.target_heading)
    if aggression_range is not None:                    # ... and how hard to throw
        critic_terms["aggression"] = ObservationTermCfg(func=mdp.target_aggression)
    if pitch_range is not None:                         # ... and the disc's nose-up pitch
        critic_terms["pitch"] = ObservationTermCfg(func=mdp.target_pitch)
    if open_loop:
        # what the robot will actually run is an open-loop joint CSV: the actor sees no sensors
        actor_terms = {
            "phase": ObservationTermCfg(func=mdp.throw_phase),
            "ref_preview": ObservationTermCfg(func=mdp.reference_preview_cmd),
            "commanded": ObservationTermCfg(func=mdp.commanded_state),
            "actions": ObservationTermCfg(func=last_action),
        }
        if heading_range is not None:
            actor_terms["heading"] = ObservationTermCfg(func=mdp.target_heading)
        if aggression_range is not None:
            actor_terms["aggression"] = ObservationTermCfg(func=mdp.target_aggression)
        if pitch_range is not None:
            actor_terms["pitch"] = ObservationTermCfg(func=mdp.target_pitch)
        critic_terms["commanded"] = ObservationTermCfg(func=mdp.commanded_state)
        critic_terms["privileged"] = ObservationTermCfg(func=mdp.privileged_grip)
    else:
        actor_terms = dict(critic_terms)
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
        "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    }

    actions = {"throw": mdp.ThrowActionCfg(robot=robot, reference_ids=reference_ids, deterministic=play,
                                           vel_cap=vel_cap, accel_cap=accel_cap, heading_range=heading_range,
                                           heading_fixed=heading_play, release_delay_range=release_delay_range,
                                           release_latency_range=release_latency_range,
                                           reference_files=reference_files, residual_scale=residual_scale,
                                           aggression_range=aggression_range, aggression_fixed=aggression_play,
                                           grip_strength=grip_strength, joint_max_override=joint_max_override,
                                           pitch_range=pitch_range, pitch_fixed=pitch_play,
                                           **({"speed_range": speed_range} if speed_range else {}),
                                           **({"release_shift_s": release_shift_s} if release_shift_s else {}),
                                           goal=mdp.ThrowGoal(**{k: v for k, v in (("speed", goal_speed), ("spin", goal_spin),
                                                                                      ("weights", goal_weights)) if v})
                                           )}

    events = {
        # fixed-base robots are wrapped in a mocap body: this places it at each env origin. In training the
        # posture varies too: lift / mount height +-3 cm, torso / mount tilt +-2 deg (roll, pitch) and the
        # robot not facing exactly forward (yaw +-2 deg). The throw direction is scored in the world frame.
        "reset_base": EventTermCfg(func=mjlab_events.reset_root_state_uniform, mode="reset",
                                   params={"pose_range": {} if play else POSTURE_RANGE, "velocity_range": {},
                                           "asset_cfg": SceneEntityCfg("robot")}),
        # --- the arm: system ID is never exact
        # link inertia +-15 % (mass and inertia together, physically consistent)
        "arm_inertia": EventTermCfg(func=dr.pseudo_inertia, mode="startup",
                                    params={"alpha_range": (-0.15, 0.15),
                                            "asset_cfg": SceneEntityCfg("robot", body_names=prof.inertia_bodies)}),
        # servo stiffness/damping +-20 %
        "arm_gains": EventTermCfg(func=mdp.pd_servo_gains, mode="startup",
                                  params={"kp_range": (0.8, 1.2), "kd_range": (0.8, 1.2)}),
        # joint friction and viscous damping (gearbox), unknown until measured
        "arm_friction": EventTermCfg(func=dr.joint_friction, mode="startup",
                                     params={"ranges": JOINT_FRICTION[robot], "asset_cfg": arm}),
        "arm_damping": EventTermCfg(func=dr.joint_damping, mode="startup",
                                    params={"ranges": (0.5, 2.0), "operation": "scale", "asset_cfg": arm}),
        # --- the frisbee: 115-200 g (mass and inertia scale by e^(2 alpha)); discs differ, and so do throws
        "frisbee_mass": EventTermCfg(func=dr.pseudo_inertia, mode="startup",
                                     params={"alpha_range": (-0.215, 0.07),
                                             "asset_cfg": SceneEntityCfg("frisbee", body_names=("frisbee",))}),
        # --- the grip, every episode: where on the rim (+-1 cm deeper/shallower, +-2 cm along the rim),
        # tilted up to 5 deg in the fingers; grip strength measured (twist 0.8, bend 1.1 N m), so +-25 % friction,
        # 0.8-1.05x squeeze, 0.85-1.15x pads: twist capacity ~0.4-1.2 N m
        "grip": EventTermCfg(func=mdp.grip_randomization, mode="reset",
                             params={"depth_m": (-0.01, 0.01), "along_rim_m": (-0.02, 0.02), "tilt_deg": 5.0,
                                     "mu_scale": (0.75, 1.25), "squeeze_scale": (0.8, 1.05),
                                     "pad_scale": (0.85, 1.15)}),
    }
    if play:
        events = {"reset_base": events["reset_base"]}

    # robot entity; in training the joint position commands arrive 0-30 ms late (mostly constant, sometimes
    # jumping: like a ROS 2 controller over a bus). Joint calibration offsets: ThrowActionCfg.joint_offset_rad.
    robot_entity = prof.robot_cfg()
    if not play:
        acts = tuple(dataclasses.replace(a, delay_min_lag=command_lag_steps[0], delay_max_lag=command_lag_steps[1],
                                         delay_hold_prob=0.95)
                     for a in robot_entity.articulation.actuators)
        robot_entity = dataclasses.replace(
            robot_entity, articulation=dataclasses.replace(robot_entity.articulation, actuators=acts))

    # the throwing arm must not hit the robot's own body
    sensors = tuple(
        ContactSensorCfg(name=f"arm_contact_{i}",
                         primary=ContactMatch(mode="subtree", pattern=prof.contact_primary, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern=sec, entity="robot"),
                         fields=("found",), reduce="none", num_slots=1)
        for i, sec in enumerate(prof.contact_secondary)) if prof.contact_primary else ()
    sensor_names = tuple(sn.name for sn in sensors)

    rewards = {
        "release_quality": RewardTermCfg(func=mdp.release_quality, weight=10.0),
        "release_shaping": RewardTermCfg(func=mdp.release_shaping, weight=1.0),
        "track_reference": RewardTermCfg(func=mdp.track_reference, weight=0.5,
                                         params={"std": 0.3, "n_joints": imitate_joints}),
        "action_rate": RewardTermCfg(func=mjlab_rewards.action_rate_l2, weight=-0.01),
        "joint_limits": RewardTermCfg(func=mjlab_rewards.joint_pos_limits, weight=-5.0, params={"asset_cfg": arm}),
        "joint_vel_excess": RewardTermCfg(func=mdp.arm_velocity_excess, weight=-1.0),
        "grip_margin": RewardTermCfg(func=mdp.grip_margin, weight=-2.0, params={"threshold": 0.7}),
        "grip_slip": RewardTermCfg(func=mdp.grip_slip, weight=-3.0),
    }
    if limit_reaction:
        effort = [a.effort_limit for a in robot_entity.articulation.actuators[:5]]
        rewards["shoulder_torque"] = RewardTermCfg(func=mdp.joint_torque_excess, weight=-2.0,
                                                   params={"limits": tuple(effort), "threshold": 0.7})
        rewards["arm_reaction"] = RewardTermCfg(func=mdp.arm_reaction, weight=-0.05)
    if sensor_names:
        rewards["body_contact"] = RewardTermCfg(func=mdp.body_contact, weight=-2.0,
                                                params={"sensor_names": sensor_names})

    terminations = {
        "time_out": TerminationTermCfg(func=mjlab_terms.time_out, time_out=True),
        "thrown": TerminationTermCfg(func=mdp.thrown, params={"after_s": 0.1}),
    }

    # start by imitating the human throw, then let the throw quality take over
    curriculum = {
        "track_reference": CurriculumTermCfg(
            func=manipulation_mdp.reward_curriculum,
            params={"reward_name": "track_reference",
                    "stages": [{"step": 0, "weight": 0.5}, {"step": 300 * 32, "weight": 0.2},
                               {"step": 800 * 32, "weight": 0.05}]},
        ),
        # friction: a firmer grip than the real one at first, the real one from iteration 600
        "grip_capacity": CurriculumTermCfg(
            func=mdp.grip_capacity_curriculum,
            params={"stages": [{"step": it * 32, "scale": sc} for it, sc in
                               (grip_curriculum or ((0, 3.0), (200, 2.0), (400, 1.4), (600, 1.0)))]},
        ),
    }
    if not imitation:                                   # maximise the throw, don't track the reference
        rewards.pop("track_reference")
        curriculum.pop("track_reference")
    if play:
        curriculum = {}

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            num_envs=1,
            env_spacing=2.5,
            entities={"robot": robot_entity, "frisbee": frisbee_cfg()},
            sensors=sensors,
            spec_fn=make_grip_weld(prof.gripper_body, prof.grip_rel_pos, prof.grip_rel_quat),
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                            body_name=prof.viewer_body, distance=2.5, elevation=-10.0, azimuth=-120.0),
        sim=SimulationCfg(
            nconmax=40, njmax=300,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,               # 50 Hz policy, like streaming /dual_arm_forward_position_controller
        episode_length_s=7.0,
    )
