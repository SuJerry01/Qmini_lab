# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Manager-based config for the Qmini BIRL locomotion task — the full faithful RoboTamer gait.

Reconstructs the RoboTamer4Qmini BIRL gait in the manager-based workflow:
- ``QminiBirlAction`` (phase oscillator + incremental joint targets, with smoothness histories),
- a single 43-d ``policy_obs`` term → 3-frame stack = deploy-faithful 129 layout,
- a ``UniformVelocityCommand`` (static cohort),
- the full 30-term faithful reward set (exp rewards w/ command-speed coefficients), and
- contact ON (the flattened ``q1.usd`` makes ``ContactSensor`` work).

``QminiBirlEnvCfg`` is the canonical task ``Template-Qmini-Walk-1kHz-v0``.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

from qmini_tasks.assets.qmini import QMINI_CFG, QMINI_FOOT_BODIES

from . import mdp

OBS_CLIP = (-3.0, 3.0)
_FOOT = SceneEntityCfg("contact_forces", body_names=list(QMINI_FOOT_BODIES))  # L,R (see rewards.py note)


# ============================================================ Scene
@configclass
class QminiSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1)
    robot = QMINI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # contact ON: q1.usd is flattened so ContactSensor `/Robot/.*` matches the foot/hip/knee bodies.
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2000.0),
    )


# ============================================================ Commands / Actions / Observations
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        # Tamer sub-threshold zeroing: zero the whole cmd if ||[vx,yaw]||<0.15, and each component if
        # |v|<0.15 — kills the "track velocity with gait rewards gated OFF" shuffle cohort.
        # STRING class_type: lazy-resolved post-app; a direct import pulls pxr into hydra cfg resolution
        # and corrupts kit startup.
        class_type="qmini_tasks.tasks.manager_based.qmini_birl.mdp.commands:QminiUniformVelocityCommand",
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),          # RoboTamer resampling_time = 5 s (config/Base.py:121)
        # RoboTamer ranges (config/Base.py:124-126); ~2% standing cohort teaches holding a pose in place.
        rel_standing_envs=0.02,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.7), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    birl_action = mdp.QminiBirlActionCfg(asset_name="robot")  # 12 = 2 phase-freq + 10 incremental joints


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # SINGLE 43-d term → 3-frame stack = deploy-faithful 129 (per-frame layout). clip on the term.
        obs = ObsTerm(func=mdp.policy_obs, clip=OBS_CLIP)

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False
            self.history_length = 3
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(PolicyCfg):
        # Tamer asymmetric critic (birl_task.py:211-238): delayed 43 (inherited) + a 64-dim fresh frame
        # (cmd errors, true lin-vel, un-delayed proprioception, net_out, foot/base ground truth).
        privileged = ObsTerm(func=mdp.critic_privileged, params={"sensor_cfg": _FOOT})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================ Terminations / Events
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    base_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.20})
    # base_link has no collision mesh → use hip/knee contacts for the undesired-contact check.
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["hip_.*", "knee_pitch_.*"]), "threshold": 1.0},
    )
    # RoboTamer training-only termination (base_task.py:289-306 jact_over): kill the degenerate
    # "stuck straight-knee crouch" — a joint whose target AND measured pos are both pinned at a limit.
    joint_at_limit = DoneTerm(func=mdp.joint_at_limit, params={"tol": 0.02})


@configclass
class EventsCfg:
    """Reset + the full RoboTamer sim2real DR set: friction / correlated mass+inertia+base-payload /
    per-reset PD-gain x torque-scale / push. The randomized obs DELAY + sensor BIAS live in the ACTION
    term (``mdp/sensor_delay.py``, cfg knobs on ``QminiBirlActionCfg``) because they tick per physics
    substep, not per event."""

    # RoboTamer randomized reset (legged_robot.py:495, random_rot=True): start from a randomized tilt +
    # base velocity + joint offset so the policy practices RECOVERING, not just balancing from rest.
    # Root offsets add to init_state (base height/quat kept).
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"roll": (-0.2, 0.2), "pitch": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
            "velocity_range": {
                # Tamer legged_robot.py:471-477: vx,vy ±0.5 / vz ±0.2 / ang ±0.5
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        # Tamer reset KICK (legged_robot.py:427-428): every episode opens with joints IN MOTION so
        # "do nothing tall" is never comfortable from frame 1. (Tamer uses a constant +2.0 rad/s via
        # ones_like; we use the idiomatic randomized ±2.0.)
        params={"position_range": (-0.1, 0.1), "velocity_range": (-2.0, 2.0)},
    )
    # randomize foot/body friction at startup — RoboTamer friction_range=[0.2,1.5] (config/Base.py:86),
    # applied to both static and dynamic friction.
    physics_material = EventTerm(
        func=randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    # PD-gain x torque-scale DR, per RESET: Tamer re-draws p_rand/d_rand/tau_gains ~U(0.8,1.2) every
    # reset (legged_robot.py:131-141) and multiplies the WHOLE torque by tau (:163-165). Factored into
    # stiffness=kp*p*tau, damping=d*tau, feedforward*tau (see mdp.randomize_pd_torque_gains).
    pd_torque_gains = EventTerm(
        func=mdp.randomize_pd_torque_gains,
        mode="reset",
        params={"gains_range": (0.8, 1.2), "torque_range": (0.8, 1.2)},
    )
    # Correlated mass/inertia DR at startup (legged_robot.py:358-376 @ :599, recomputeInertia=False):
    # ONE dm~U(0.5,1.5) on every non-base link, ONE di~U(0.5,1.5) on every link's inertia, base
    # payload ADD U(-0.6,+0.7)*m0.
    mass_inertia = EventTerm(
        func=mdp.randomize_mass_inertia_tamer,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_range": (0.5, 1.5),
            "inertia_range": (0.5, 1.5),
            "base_add_range": (-0.6, 0.7),
        },
    )
    # Tamer-faithful push every 3 s (legged_robot.py:495-508): REPLACE root velocity with bounded
    # U(-0.5r,0.5r) + SNAP orientation near-upright (rescue) + ramp r 1->1.5 over 3000 iters. The stock
    # additive push (no rescue, no bound, no ramp) is a harsher regime favouring straight-leg strategies.
    push_robot = EventTerm(
        func=mdp.push_replace_vel_and_right,
        mode="interval",
        interval_range_s=(3.0, 3.0),
        params={"base_magnitude": 0.5, "tilt": 0.2, "ramp_end_iters": 3000},
    )


# ============================================================ Rewards — the full 30-term Tamer set
@configclass
class RewardsCfg:
    """The full-Tamer 30-term reward set. The balance-gate multiply on the ~16 style terms is wired in
    ``mdp/rewards.py`` via ``_balgate``. No termination penalty — the summed step reward is floored at 0
    by :class:`QminiBirlEnv` (RoboTamer ``gym_env_wrapper.py:66``), so staying alive is never worse.
    """
    # tracking + stability + survival (always-on core; not balance-gated)
    track_lin_vel = RewTerm(func=mdp.track_lin_vel, weight=2.3)
    track_yaw = RewTerm(func=mdp.track_yaw, weight=2.5)
    base_height = RewTerm(func=mdp.base_height_exp, weight=1.0, params={"target_height": 0.45})
    balance = RewTerm(func=mdp.balance_factor, weight=1.5)
    lateral_vel = RewTerm(func=mdp.lateral_vel, weight=0.7)
    vertical_vel = RewTerm(func=mdp.vertical_vel, weight=0.6)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy, weight=0.6)
    twist = RewTerm(func=mdp.twist, weight=2.5)
    alive = RewTerm(func=mdp.is_alive, weight=0.3)
    # gait phasing (anti-phase alternating legs)
    foot_phase = RewTerm(func=mdp.foot_phase_antiphase, weight=-0.3)
    # foot / contact shaping (needs the contact sensor)
    foot_clr = RewTerm(func=mdp.foot_clearance, weight=1.0, params={"sensor_cfg": _FOOT})
    foot_supt = RewTerm(func=mdp.foot_support, weight=0.7, params={"sensor_cfg": _FOOT})
    foot_heit = RewTerm(func=mdp.foot_height, weight=0.7, params={"sensor_cfg": _FOOT})
    foot_sft = RewTerm(func=mdp.foot_soft, weight=2.7, params={"sensor_cfg": _FOOT})
    feet_frc = RewTerm(func=mdp.feet_contact_frc, weight=0.001, params={"sensor_cfg": _FOOT})
    foot_acc = RewTerm(func=mdp.foot_acc, weight=0.05)
    feet_py = RewTerm(func=mdp.foot_py, weight=0.5)
    # smoothness (2nd-difference jerk on joint targets / network output)
    act_smo = RewTerm(func=mdp.action_smooth, weight=1.5)
    net_smo = RewTerm(func=mdp.net_out_smooth, weight=0.001)
    pmf = RewTerm(func=mdp.pmf_smooth, weight=0.03)
    net_out_val = RewTerm(func=mdp.net_out_val, weight=0.0001)
    # posture & effort regularization
    act_const = RewTerm(func=mdp.action_constraint, weight=0.2)
    jnt_pos_err = RewTerm(func=mdp.joint_pos_error, weight=0.2)
    jnt_vel = RewTerm(func=mdp.joint_vel, weight=0.003)
    joint_tor = RewTerm(func=mdp.joint_torque, weight=0.001)
    leg_width = RewTerm(func=mdp.leg_width, weight=0.5)
    base_acc = RewTerm(func=mdp.base_acc, weight=0.1)
    sa_const = RewTerm(func=mdp.sa_const, weight=0.1)
    foot_slip = RewTerm(func=mdp.foot_slip, weight=0.5)
    foot_vz = RewTerm(func=mdp.foot_vz, weight=0.2)


# ============================================================ Environment — THE canonical full task
@configclass
class QminiBirlEnvCfg(ManagerBasedRLEnvCfg):
    """The full-Tamer Qmini walking config: RewardsCfg (30 terms, balance-gated) + the full sim2real DR
    set (EventsCfg) + Tamer-parity commands (CommandsCfg) + the RoboTamer sensor model (obs delay +
    bias, in the action term). Registered as ``Template-Qmini-Walk-1kHz-v0``. Known approximations:
    heading-frame→world foot_vel (critic/leg_width) and the torque clip-before-tau nuance
    (see mdp.randomize_pd_torque_gains)."""

    scene: QminiSceneCfg = QminiSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self) -> None:
        # control rate: dt 0.001 x decimation 15 = 0.015 s policy step (the contract)
        self.decimation = 15
        self.episode_length_s = 10.0
        # NO value bootstrap at the 10 s timeout: Tamer's timeout bootstrap is effectively dead
        # (extra_info["timeouts"] bound once at base_task.py:131 while legged_robot.py:299 rebinds
        # time_out_buf per step -> their PPO :75 never fires), so the golden gait was trained EPISODIC.
        # Finite-horizon = parity with the effective Tamer semantics.
        self.is_finite_horizon = True
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        # PhysX GPU buffers raised: an untrained policy tangles bipeds → contact explosion at scale.
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=5_000_000, gpu_max_rigid_contact_count=100_000_000)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (4.0, 0.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)


# ============================================================ Play/eval variants (NVIDIA -Play convention)
def _disable_training_randomization(cfg: QminiBirlEnvCfg) -> None:
    """Turn off training-time randomization for clean deterministic play/eval — shared by the -Play cfgs.

    Disabled: interval push, per-reset PD/torque + mass/inertia + friction DR, the action-term randomized
    sensor delay + bias, obs corruption; and a small scene. Keeps reset_root / reset_joints (needed to
    spawn) and the full reward/termination set. For a robustness view WITH disturbances, use the base task.
    """
    cfg.scene.num_envs = 4                              # fewer envs for viewing (CLI --num_envs overrides)
    cfg.observations.policy.enable_corruption = False   # no observation corruption
    cfg.events.push_robot = None                        # no interval push
    cfg.events.pd_torque_gains = None                   # no per-reset domain randomization
    cfg.events.mass_inertia = None
    cfg.events.physics_material = None
    cfg.actions.birl_action.sensor_delay = False        # no randomized sensor delay / bias
    cfg.actions.birl_action.sensor_bias_noise = False


@configclass
class QminiBirlPlayEnvCfg(QminiBirlEnvCfg):
    """1 kHz play/eval — SAME MDP as ``Template-Qmini-Walk-1kHz-v0`` with training randomization OFF.
    Registered as ``Template-Qmini-Walk-1kHz-Play-v0``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _disable_training_randomization(self)


# ============================================================ 200 Hz-physics experiment variant (A/B)
@configclass
class QminiBirl200HzEnvCfg(QminiBirlEnvCfg):
    """A/B experiment: 200 Hz physics with the deploy interface unchanged.

    ``sim.dt 0.005 × decimation 3 = control_dt 0.015 s`` — same 66.7 Hz policy interface (the C++ SDK
    contract) as the 1 kHz config; only the inner-loop physics rate changes. PD gains kept identical on
    purpose — the behavior delta IS the measurement. Registered as ``Template-Qmini-Walk-200Hz-v0``.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 3
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # sensor-delay ring buffers tick PER PHYSICS SUBSTEP. Rescale the substep-count ranges to keep
        # the same TIME delays at 5 ms substeps: joint 10-40 @1 ms -> (2, 8) @5 ms; rate/angle 20-50 -> (4, 10).
        self.actions.birl_action.delay_joint_range = (2, 8)
        self.actions.birl_action.delay_rate_range = (4, 10)
        self.actions.birl_action.delay_angle_range = (4, 10)


# ============================================================ 200 Hz play/eval variant
@configclass
class QminiBirl200HzPlayEnvCfg(QminiBirl200HzEnvCfg):
    """200 Hz play/eval — 200 Hz physics with training randomization OFF.
    Registered as ``Template-Qmini-Walk-200Hz-Play-v0``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _disable_training_randomization(self)
