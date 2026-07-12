# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Config for the Qmini BIRL locomotion task (Direct workflow).

Reconstructs the RoboTamer4Qmini sim/obs/action/reward parameters into an Isaac Lab
``DirectRLEnvCfg``. Numbers (control rate, obs/action dims, command ranges, reward weights,
PhaseModulator settings) are sourced from ``config/Base.py`` / ``config/BIRL.py`` and the
sim-to-real contract (see ``docs/RoboTamer4Qmini_codebase.md`` / ``docs/deep_dive.md``).

Follows the shipped cartpole Direct template's structure (top-level ``robot_cfg`` built in
``_setup_scene``) rather than SceneCfg auto-instantiation.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from qmini_tasks.assets.qmini import QMINI_CFG

# ---- contract dims (see deep_dive.md) ----
PURE_OBS_DIM = 43                            # single-frame "pure" observation
OBS_STACK = 3                                # 3-frame stacking
POLICY_OBS_DIM = PURE_OBS_DIM * OBS_STACK    # 129 actor inputs
NUM_ACTIONS = 12                             # 2 phase-frequency + 10 incremental joint targets
PRIV_DIM = 8                                 # privileged critic block: base lin vel(3)+base height(1)+foot z(2)+foot force(2)
CRITIC_OBS_DIM = POLICY_OBS_DIM + PRIV_DIM   # 137 (asymmetric, training-only)


@configclass
class QminiLabEnvCfg(DirectRLEnvCfg):
    """Direct-workflow config for the Qmini BIRL gait."""

    # --- control rate: dt 0.001 x decimation 15 = 0.015 s policy step (the contract) ---
    decimation = 15
    episode_length_s = 10.0

    # --- spaces (asymmetric actor/critic; built in QminiLabEnv._get_observations) ---
    action_space = NUM_ACTIONS           # 12
    observation_space = POLICY_OBS_DIM   # 129  -> obs dict key "policy"
    state_space = CRITIC_OBS_DIM         # 137  -> obs dict key "critic" (privileged superset)

    # --- simulation: 1 kHz physics, render at the policy step ---
    # PhysX GPU buffers raised: at many envs an untrained policy collapses the bipeds into tangled
    # heaps -> contact/patch counts explode past the defaults (patch 163840, contact 8.4M). TODO: once
    # dynamics are tuned (robots stay upright) these can be lowered back toward defaults.
    sim: SimulationCfg = SimulationCfg(
        dt=0.001,
        render_interval=decimation,
        physics=PhysxCfg(
            gpu_max_rigid_patch_count=5_000_000,
            gpu_max_rigid_contact_count=100_000_000,
        ),
    )

    # --- robot (built in _setup_scene, cartpole-template style) ---
    robot_cfg: ArticulationCfg = QMINI_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # whole-body contact sensor; env resolves foot/termination body ids via find_bodies
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
    )
    # Contact sensing requires a FLAT body layout. The 3.0 URDF importer nests the link prims
    # (Robot/Geometry/base_link/hip/.../ankle), which Isaac Lab's ContactSensor (`/Robot/.*`) cannot
    # match. FIXED (2026-06-19): the asset now loads a pre-FLATTENED USD (assets/q1/q1.usd, produced by
    # scripts/flatten_qmini_usd.py — flat `/Robot/<link>` like the shipped /h1) so ContactSensor
    # initializes. Verified: a contact-ON train smoke runs cleanly (no "could not find any bodies").
    # Set False only to deliberately run contact-free (foot-force obs → zeros; gait-contact reward and
    # illegal-contact termination skipped; 129/137 dims preserved either way).
    use_contact_sensor: bool = True

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # === task parameters (reconstructed from RoboTamer config/BIRL.py) ===
    # action de-scale ranges (inc_low/high_ranges): phase x2, joints x10
    phase_freq_range = (0.5, 3.5)        # per-leg frequency band
    joint_rate_range = (-15.0, 15.0)     # joint-target rate [rad/s], integrated at step_dt
    obs_clip = 3.0                       # pure-obs clip to [-3, 3]
    ang_vel_scale = 0.5
    joint_vel_scale = 0.1
    # gait
    phase_rest_frequency = 0.5
    phase_convert_phi = 1.2 * math.pi    # ~60% stance duty
    static_speed_threshold = 0.15        # ||[vx, yaw]|| below this => standing (gait obs/reward gated off)
    # command sampling (5 s resample + static cohort)
    command_vx_range = (-0.6, 1.2)
    command_yaw_range = (-1.0, 1.0)
    command_resample_time_s = 5.0
    static_command_fraction = 0.02       # fraction of envs forced to zero command (static cohort)
    # termination
    base_height_termination = 0.20       # base z below this => fell
    max_tilt = 0.7                       # |roll| or |pitch| above this [rad] => fell
    base_height_target = 0.45            # base-height reward target

    # === reward weights (RoboTamer birl_task rew_dict; faithful subset implemented in M1) ===
    rew_track_lin_vel = 2.3              # fwd_vel (track vx command)
    rew_track_yaw = 2.5                  # yaw_rat (track yaw-rate command)
    rew_base_height = 1.0                # base_heit
    rew_lateral_vel = -0.7              # suppress uncommanded vy
    rew_vertical_vel = -0.6             # suppress vz
    rew_ang_vel_xy = -0.6               # suppress roll/pitch rate
    rew_orientation = -2.5              # twist (|roll, pitch|)
    rew_foot_phase = -0.3               # anti-phase term (legs 180 deg apart)
    rew_action_smooth = -1.5            # 2nd-difference of joint targets (jerk)
    rew_freq_smooth = -0.03             # 2nd-difference of frequency outputs (pmf)
    rew_joint_vel = -0.003              # joint-velocity penalty
    rew_joint_torque = -2.5e-4          # applied-torque (effort) penalty
    rew_foot_contact = 0.7              # measured contact matching the gait swing/support mask
    rew_foot_clearance = 0.7            # swing-foot clearance
    rew_alive = 0.3                     # constant alive bonus
    rew_termination = -10.0             # failure penalty

    def __post_init__(self) -> None:
        self.viewer.eye = (4.0, 0.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
