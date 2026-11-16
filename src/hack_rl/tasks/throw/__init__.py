"""Frisbee throw tasks: the Panther right arm, and the Franka Panda as an off-the-shelf comparison.

Panther variants: see the comments in VARIANTS (dr6 onwards; dr1-dr5 removed 2026-11-15).
"""
from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import throw_env_cfg
from .rl_cfg import throw_ppo_runner_cfg

TASK_IDS = {"panther": "Hack-Throw-Frisbee-Panther", "franka": "Hack-Throw-Frisbee-Franka"}
TASK_ID = TASK_IDS["panther"]
VARIANTS = {
    # dr6: like dr5 but "maximise, not track" (user, after robot test 4): no practical velocity/acceleration
    # caps (10 / 15 rad/s, 100 / 200 rad/s^2; the sim's torque limits 31 / 11 N m and joint limits decide), speed
    # and spin targets 10 m/s and 30 rad/s so the reward never saturates, more weight on speed and spin, no
    # imitation reward, bigger shoulder/elbow residuals, free wrist
    "Hack-Throw-Frisbee-Panther-Max": dict(vel_cap=(10.0,) * 5 + (15.0,) * 3, accel_cap=(100.0,) * 5 + (200.0,) * 3,
                                           heading_range=(-35.0, 5.0), heading_play=-15.0,
                                           release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                           command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                           goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                         "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                           reference_files=("enlarged/joints_trajectory_shoulder_rel_2.csv",
                                                            "enlarged/joints_trajectory_shoulder_rel_3.csv"),
                                           residual_scale=(0.6, 0.6, 0.6, 0.6, 1.2, 1.5, 0.9, 1.6), imitation=False,
                                           # dr6b: with no caps, a 3x warm-up grip taught throws the real fingers
                                           # can't hold (slip 95 % when it dropped to 2x): mild warm-up only
                                           grip_curriculum=((0, 1.5), (100, 1.2), (200, 1.0))),
    # dr7: Max + the user's grip ("firm at position 1.0": friction x3), acceleration limited to what exploration
    # noise tolerates (30 / 80 rad/s^2, still ~2x fast2), speed unlimited, and an aggressiveness knob: each
    # episode asks for a*10 m/s and a*30 rad/s spin, a in 0.3..1 observed; export with --aggression a
    "Hack-Throw-Frisbee-Panther-Aggro": dict(vel_cap=(10.0,) * 5 + (15.0,) * 3, accel_cap=(30.0,) * 5 + (80.0,) * 3,
                                             heading_range=(-35.0, 5.0), heading_play=-15.0,
                                             release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                             command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                             goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                           "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                             reference_files=("enlarged/joints_trajectory_shoulder_rel_2.csv",
                                                              "enlarged/joints_trajectory_shoulder_rel_3.csv"),
                                             residual_scale=(0.6, 0.6, 0.6, 0.6, 1.2, 1.5, 0.9, 1.6), imitation=False,
                                             grip_strength=3.0, grip_curriculum=((0, 1.0),),
                                             aggression_range=(0.3, 1.0), aggression_play=0.7),
    # dr8: reference preprocessing v2 (hack_rl.prepare_references_v2): the human throw at the HUMAN's timing (the
    # HEADBands slow-down undone), arm ra0-ra4 as tracked, release at peak hand speed, wrist ra5-ra7 flattened
    # + ra7 flick seed. Playback 0.5-1.5x around human timing, release shift +-0.1 s, arm residuals +-0.4 rad,
    # wrist moderate (user: the wrist has a safety limit), joints <= 10 rad/s, 30 / 80 rad/s^2, grip x3,
    # aggressiveness knob, heading -60..+10 deg (ref3 releases at -47 deg)
    "Hack-Throw-Frisbee-Panther-Human": dict(vel_cap=(10.0,) * 8, accel_cap=(30.0,) * 5 + (80.0,) * 3,
                                             heading_range=(-60.0, 10.0), heading_play=-25.0,
                                             release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                             command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                             goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                           "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                             reference_files=("human_speed/joints_trajectory_shoulder_rel_2.csv",
                                                              "human_speed/joints_trajectory_shoulder_rel_3.csv"),
                                             residual_scale=(0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.6, 1.0), imitation=False,
                                             grip_strength=3.0, grip_curriculum=((0, 1.0),),
                                             aggression_range=(0.3, 1.0), aggression_play=0.5,
                                             speed_range=(0.5, 1.5), release_shift_s=0.1),
    # dr9: same training as dr8, on reference preprocessing v3 (hack_rl.prepare_references_v3: IK retarget of the
    # tracked elbow/wrist, elbow >= ~45 deg bent, >= 3 cm off the torso, human timing; releases at -15 deg)
    "Hack-Throw-Frisbee-Panther-IK": dict(vel_cap=(10.0,) * 8, accel_cap=(30.0,) * 5 + (80.0,) * 3,
                                          heading_range=(-45.0, 15.0), heading_play=-15.0,
                                          release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                          command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                          goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                        "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                          reference_files=("ik/joints_trajectory_shoulder_rel_2.csv",
                                                           "ik/joints_trajectory_shoulder_rel_3.csv"),
                                          residual_scale=(0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.6, 1.0), imitation=False,
                                          grip_strength=3.0, grip_curriculum=((0, 1.0),),
                                          aggression_range=(0.3, 1.0), aggression_play=0.5,
                                          speed_range=(0.5, 1.5), release_shift_s=0.1),
    # dr10: dr9 + the elbow kept bent: commanded ra4 <= -0.6 rad (~35 deg). dr9 straightened it to -0.06 rad (3 deg)
    # with its +-0.4 residual because a straight arm throws a bit faster; the user rejects the extended arm
    "Hack-Throw-Frisbee-Panther-IK-Bent": dict(vel_cap=(10.0,) * 8, accel_cap=(30.0,) * 5 + (80.0,) * 3,
                                               heading_range=(-45.0, 15.0), heading_play=-15.0,
                                               release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                               command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                               goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                             "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                               reference_files=("ik/joints_trajectory_shoulder_rel_2.csv",
                                                                "ik/joints_trajectory_shoulder_rel_3.csv"),
                                               residual_scale=(0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.6, 1.0), imitation=False,
                                               grip_strength=3.0, grip_curriculum=((0, 1.0),),
                                               aggression_range=(0.3, 1.0), aggression_play=0.5,
                                               speed_range=(0.5, 1.5), release_shift_s=0.1,
                                               joint_max_override={4: -0.6}),
    # dr10: reference v4 stroke-only (prepare_references_v3 --chest-frame --stroke-only -> data/reference/stroke):
    # the arm retargeted in the human chest frame (IK error 0.2 cm), starting at the deepest backswing, back-to-front
    # stroke, release ~9 m/s at the path's own heading (ref2 -65, ref3 -55 deg: turn the base ~60 deg left).
    # Target heading -85..-35 deg (each throw's own direction +-20), elbow kept bent (ra4 <= -0.6), dr9's settings
    "Hack-Throw-Frisbee-Panther-Stroke": dict(vel_cap=(10.0,) * 8, accel_cap=(30.0,) * 5 + (80.0,) * 3,
                                              heading_range=(-85.0, -35.0), heading_play=-60.0,
                                              release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                              command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                              goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                            "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                              reference_files=("stroke/joints_trajectory_shoulder_rel_2.csv",
                                                               "stroke/joints_trajectory_shoulder_rel_3.csv"),
                                              residual_scale=(0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.6, 1.0), imitation=False,
                                              grip_strength=3.0, grip_curriculum=((0, 1.0),),
                                              aggression_range=(0.3, 1.0), aggression_play=0.5,
                                              speed_range=(0.5, 1.5), release_shift_s=0.1,
                                              joint_max_override={4: -0.6}),
    # dr11: dr10 + a commanded PITCH (nose up, 0..15 deg, observed; export --pitch) next to the aggressiveness:
    # the "flat" and "angle of attack" reward terms become one attitude term towards the commanded disc normal
    "Hack-Throw-Frisbee-Panther-Stroke-Pitch": dict(vel_cap=(10.0,) * 8, accel_cap=(30.0,) * 5 + (80.0,) * 3,
                                                    heading_range=(-85.0, -35.0), heading_play=-60.0,
                                                    release_delay_range=(0.15, 0.21), release_latency_range=(0.0, 0.02),
                                                    command_lag_steps=(0, 2), goal_speed=10.0, goal_spin=30.0,
                                                    goal_weights={"speed_dir": 0.35, "flat": 0.15, "spin": 0.25,
                                                                  "elevation": 0.08, "aoa": 0.09, "no_wobble": 0.08},
                                                    reference_files=("stroke/joints_trajectory_shoulder_rel_2.csv",
                                                                     "stroke/joints_trajectory_shoulder_rel_3.csv"),
                                                    residual_scale=(0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.6, 1.0), imitation=False,
                                                    grip_strength=3.0, grip_curriculum=((0, 1.0),),
                                                    aggression_range=(0.3, 1.0), aggression_play=0.8,
                                                    pitch_range=(0.0, 15.0), pitch_play=5.0,
                                                    speed_range=(0.5, 1.5), release_shift_s=0.1,
                                                    joint_max_override={4: -0.6}),
}
ROBOT_OF_TASK = {v: k for k, v in TASK_IDS.items()} | {k: "panther" for k in VARIANTS}

for _robot, _task in TASK_IDS.items():
    register_mjlab_task(
        task_id=_task,
        env_cfg=throw_env_cfg(_robot),
        play_env_cfg=throw_env_cfg(_robot, play=True),
        rl_cfg=throw_ppo_runner_cfg(_robot),
    )
for _task, _kw in VARIANTS.items():
    register_mjlab_task(
        task_id=_task,
        env_cfg=throw_env_cfg("panther", **_kw),
        play_env_cfg=throw_env_cfg("panther", play=True, **_kw),
        rl_cfg=throw_ppo_runner_cfg("panther"),
    )
