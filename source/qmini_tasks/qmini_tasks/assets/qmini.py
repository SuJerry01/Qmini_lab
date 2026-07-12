# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""ArticulationCfg for the Unitree Qmini biped (q1 URDF), ported from RoboTamer4Qmini.

Joint order is the sim-to-real contract: the 10 actuated joints use the deploy order
hip_yaw, hip_roll, hip_pitch, knee, ankle (left then right). The env resolves indices via
find_joints(QMINI_JOINT_ORDER, preserve_order=True) and remaps every 10-vector back to it.

Gains here are the SIMULATION PD gains (high) from RoboTamer config/Base.py pd_gains; the low
robot motor kp/kd live only in the deploy config.yaml and must not be copied here. Limits/effort/
velocity come from the URDF (hip_roll: effort 60 / vel 10; the rest: 20 / 30).

Loads a pre-converted, flattened USD (q1.usd) via UsdFileCfg, not an on-the-fly URDF import: the 3.0
URDF importer nests link prims and breaks the foot ContactSensor, so the USD is flattened to flat
/Robot/<link> prims. Original RoboTamer URDF + meshes kept read-only under assets/q1-tamer/.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# --- asset paths (asset copied from the read-only RoboTamer4Qmini reference) ---
QMINI_ASSET_DIR = os.path.join(os.path.dirname(__file__), "q1")
QMINI_URDF_PATH = os.path.join(QMINI_ASSET_DIR, "urdf", "q1.urdf")
# Pre-converted, flattened USD (scripts/flatten_qmini_usd.py): flat /Robot/<link> prims so the foot
# ContactSensor's /Robot/.* match works (the 3.0 URDF importer nests link prims and breaks it).
QMINI_USD_PATH = os.path.join(QMINI_ASSET_DIR, "q1.usd")

# --- the contract: 10 joints in deploy order (L then R) ---
QMINI_JOINT_ORDER = [
    "hip_yaw_l", "hip_roll_l", "hip_pitch_l", "knee_pitch_l", "ankle_pitch_l",
    "hip_yaw_r", "hip_roll_r", "hip_pitch_r", "knee_pitch_r", "ankle_pitch_r",
]
# Foot bodies (the only leg links with collision geometry in the URDF).
QMINI_FOOT_BODIES = ["ankle_pitch_l", "ankle_pitch_r"]
# Bodies whose contact terminates an episode. NOTE: base_link has NO collision mesh in the URDF,
# so base contact cannot be sensed — the env uses base height instead for the "fell over" check.
QMINI_TERMINATION_BODIES = ["hip_yaw_.*", "hip_roll_.*", "hip_pitch_.*", "knee_pitch_.*"]

# ref_joint_pos / standing reference (RoboTamer config/BIRL.py). The obs subtracts this and the
# "stand" reset returns here.
QMINI_REF_JOINT_POS = [0.4, -0.1, -1.5, 1.0, -1.3, -0.4, 0.1, 1.5, -1.0, 1.3]
REF_JOINT_POS = dict(zip(QMINI_JOINT_ORDER, QMINI_REF_JOINT_POS))

# Joint POSITION limits = sim-to-real contract (deploy config.yaml act_pos_low/high), deploy joint
# order. The action term applies these (clip + write_joint_position_limit_to_sim) instead of the URDF
# limits, which the 3.0 importer imported sign-flipped and would clip the crouch ref pose.
QMINI_POS_LIMIT_LOW  = [-0.1, -0.3, -2.1, 0.0, -2.5,  -0.7, -0.6, 0.0, -2.1, 0.0]
QMINI_POS_LIMIT_HIGH = [ 0.7,  0.6,  0.0, 2.1,  0.0,   0.1,  0.3, 2.1,  0.0, 2.5]


QMINI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=QMINI_USD_PATH,         # pre-converted + FLATTENED USD (flat link prims like /h1/<link>)
        activate_contact_sensors=True,   # apply the PhysX contact-report API so the foot ContactSensor works
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Free floating base (a biped must float to walk). The world->base fixed root_joint is
            # deactivated in the USD by scripts/flatten_qmini_usd.py; this flag documents the intent.
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Standing base height = RoboTamer init_state.pos (config/Base.py) = 0.45 (base_heit reward target).
        pos=(0.0, 0.0, 0.45),
        joint_pos=REF_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    # Soft limit margin used for the learning-side action clip (physics still enforces hard limits).
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # SIM-side PD gains: stiffness (kp) from RoboTamer config/Base.py pd_gains; damping (kd) = 1.0
        # uniform. kd is the velocity coefficient (RoboTamer's d_gains_rand, DR'd to ~1.0), NOT the config
        # `damping` dict {0.3,2.5,0.3,0.5,0.25} — that dict is a constant torque bias applied as effort
        # feedforward in actions.py. Nominal kp is mirrored in mdp/qmini_events.py _NOMINAL_KP (per-reset
        # PD DR) — keep the two in lockstep.
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["hip_yaw_.*"], effort_limit_sim=20.0, velocity_limit_sim=30.0,
            stiffness=55.0, damping=1.0,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["hip_roll_.*"], effort_limit_sim=60.0, velocity_limit_sim=10.0,
            stiffness=105.0, damping=1.0,
        ),
        "hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=["hip_pitch_.*"], effort_limit_sim=20.0, velocity_limit_sim=30.0,
            stiffness=75.0, damping=1.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["knee_pitch_.*"], effort_limit_sim=20.0, velocity_limit_sim=30.0,
            stiffness=45.0, damping=1.0,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=["ankle_pitch_.*"], effort_limit_sim=20.0, velocity_limit_sim=30.0,
            stiffness=30.0, damping=1.0,
        ),
    },
)
"""Qmini biped articulation. Joints addressed in deploy order via :data:`QMINI_JOINT_ORDER`."""
