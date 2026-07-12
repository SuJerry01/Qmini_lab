# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Faithful reward terms for the Qmini BIRL gait (RoboTamer BIRLTask.reward terms).

Each function reproduces a RoboTamer reward term — exp rewards with command-speed-dependent
coefficients, the ``balance`` gate, and ``static_flag`` gating. Functions return the UNWEIGHTED
per-env value ``(N,)``; the manager applies ``weight * value * step_dt``.

Conventions:
- ``lin_vel_x_norm = clip(|cmd_vx|, 0.3, 2) + 0.2`` — RoboTamer normaliser scaling many coefficients.
- ``static_flag = (||[vx,yaw]|| >= 0.15)`` — gates gait/foot/style terms off while standing.
- ``balance ∈ [0.5,1]`` multiplies the ~15 style terms (NOT the tracking core); each gated term applies
  it explicitly in its own body (see the balance-gate section).
- Foot data is read in deploy (L,R) order via the action term's ``foot_body_ids`` + a ``preserve_order``
  contact SceneEntityCfg, so it lines up with the per-leg support/swing masks.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

# URDF effort limits in deploy joint order (hip_roll 60, the rest 20).
_TORQUE_LIMIT = (20.0, 60.0, 20.0, 20.0, 20.0, 20.0, 60.0, 20.0, 20.0, 20.0)


# ============================================================ helpers
def _cmd(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``[vx, yaw]`` command — shape ``(N,2)``."""
    return env.command_manager.get_command(command_name)[:, [0, 2]]


def _lin_vel_x_norm(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``clip(|vx_cmd|, 0.3, 2) + 0.2`` — shape ``(N,)``."""
    return torch.clip(_cmd(env, command_name)[:, 0].abs(), 0.3, 2.0) + 0.2


def _yaw_rate_norm(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return torch.clip(_cmd(env, command_name)[:, 1].abs(), 0.3, 1.5) + 0.2


def _static_flag(env: ManagerBasedRLEnv, command_name: str, threshold: float = 0.15) -> torch.Tensor:
    """``1`` iff commanded, else ``0`` — shape ``(N,)``."""
    return (torch.norm(_cmd(env, command_name), dim=1) >= threshold).float()


def _euler_rp(asset: "Articulation") -> tuple[torch.Tensor, torch.Tensor]:
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return wrap_to_pi(roll), wrap_to_pi(pitch)


def _balance(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``balance_rew ∈ [0.5,1]`` = ``0.5*(base_heit*exp(-k*||roll,pitch||)+1)`` — shape ``(N,)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    z = asset.data.root_pos_w.torch[:, 2]
    roll, pitch = _euler_rp(asset)
    lvxn = _lin_vel_x_norm(env, command_name)
    base_heit = torch.exp(-70.0 * torch.square(z - 0.45))
    k = torch.clip(5.0 / lvxn, 2.0, 8.0)
    tilt = torch.sqrt(roll * roll + pitch * pitch + 1e-8)
    return 0.5 * (base_heit * torch.exp(-k * tilt) + 1.0)


def _foot_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Per-leg foot contact force magnitude ``(N,2)`` (L,R). Zeros if contact sensing is disabled."""
    if "contact_forces" not in env.scene.sensors:
        return torch.zeros(env.num_envs, 2, device=env.device)
    sensor = env.scene.sensors["contact_forces"]
    return torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :], dim=-1)


def _foot_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str) -> torch.Tensor:
    """Per-leg foot linear velocity in world frame ``(N,2,3)`` (L,R)."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    return asset.data.body_lin_vel_w.torch[:, term.foot_body_ids, :]


def _foot_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, action_name: str) -> torch.Tensor:
    """Per-leg SOLE clearance above flat ground ``(N,2)`` (L,R).

    The ankle body origin sits ~0.095 m above the sole — RoboTamer subtracts 0.1 before any foot term
    so a planted foot reads ~0 clearance (else foot_heit sees phantom swing).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    return asset.data.body_pos_w.torch[:, term.foot_body_ids, 2] - 0.1


# ============================================================ tracking + stability
def track_lin_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``fwd_vel``: ``exp(-clip(5/lvxn,2,10)*(vx_cmd - vx)^2)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b.torch[:, 0]
    lvxn = _lin_vel_x_norm(env, command_name)
    k = torch.clip(5.0 / lvxn, 2.0, 10.0)
    return torch.exp(-k * torch.square(_cmd(env, command_name)[:, 0] - vx))


def track_yaw(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``yaw_rat``: ``exp(-clip(2/yawn,2,6)*(yaw_cmd - wz)^2)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    wz = asset.data.root_ang_vel_b.torch[:, 2]
    k = torch.clip(2.0 / _yaw_rate_norm(env, command_name), 2.0, 6.0)
    return torch.exp(-k * torch.square(_cmd(env, command_name)[:, 1] - wz))


def base_height_exp(env: ManagerBasedRLEnv, target_height: float = 0.45, sharpness: float = 70.0,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``base_heit``: ``exp(-sharpness*(z - target)^2)``."""
    z = env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2]
    return torch.exp(-sharpness * torch.square(z - target_height))


def lateral_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``lateral_vel``: ``exp(-clip(5/lvxn,3,15)*vy^2) - 0.6/lvxn*|vy|*static``."""
    asset: Articulation = env.scene[asset_cfg.name]
    vy = asset.data.root_lin_vel_b.torch[:, 1]
    lvxn = _lin_vel_x_norm(env, command_name)
    sf = _static_flag(env, command_name)
    main = torch.exp(-torch.clip(5.0 / lvxn, 3.0, 15.0) * vy * vy)
    return main - 0.6 / lvxn * vy.abs() * sf


def vertical_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``vertical_vel``: ``exp(-clip(5/lvxn,2,10)*vz^2) - 0.2/lvxn*||[vy,vz]||*static``."""
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b.torch
    lvxn = _lin_vel_x_norm(env, command_name)
    sf = _static_flag(env, command_name)
    main = torch.exp(-torch.clip(5.0 / lvxn, 2.0, 10.0) * v[:, 2] * v[:, 2])
    return main - 0.2 / lvxn * torch.norm(v[:, 1:], dim=1) * sf


def ang_vel_xy(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
               asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``ang_vel``: ``exp(-clip(2/lvxn,0.7,6)*||w_xy||^2)``."""
    asset: Articulation = env.scene[asset_cfg.name]
    wxy = asset.data.root_ang_vel_b.torch[:, :2]
    k = torch.clip(2.0 / _lin_vel_x_norm(env, command_name), 0.7, 6.0)
    return torch.exp(-k * torch.sum(wxy * wxy, dim=1))


def twist(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``twist``: ``-||roll, pitch||``."""
    roll, pitch = _euler_rp(env.scene[asset_cfg.name])
    return -torch.sqrt(roll * roll + pitch * pitch + 1e-8)


# ============================================================ gait phasing
def foot_phase_antiphase(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                         action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``foot_phase``: ``(sinL+sinR)^2+(cosL+cosR)^2`` (=0 anti-phase), static-gated + balance.

    Use a NEGATIVE weight to enforce an alternating gait.
    """
    sc = env.action_manager.get_term(action_name).phase_sin_cos  # [sinL,sinR,cosL,cosR]
    val = torch.square(sc[:, 0] + sc[:, 1]) + torch.square(sc[:, 2] + sc[:, 3])
    return val * _static_flag(env, command_name) * _balance(env)


def feet_air_time_biped(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                        threshold: float = 0.4) -> torch.Tensor:
    """Biped single-stance air-time reward (Isaac Lab H1/G1 `feet_air_time_positive_biped` shape).

    Rewards the time a foot spends in its current mode (air OR contact) while exactly one foot is down,
    forcing real alternating swings. POSITIVE weight; zeroed when commanded to stand (||[vx,yaw]||<0.1).
    Needs the ContactSensor with ``track_air_time=True``.
    """
    if "contact_forces" not in env.scene.sensors:
        return torch.zeros(env.num_envs, device=env.device)
    sensor = env.scene.sensors["contact_forces"]
    ids = sensor_cfg.body_ids
    air = sensor.data.current_air_time.torch[:, ids]        # (N,2)
    contact = sensor.data.current_contact_time.torch[:, ids]
    in_contact = contact > 0.0
    in_mode_time = torch.where(in_contact, contact, air)          # time in current mode
    single_stance = in_contact.int().sum(dim=1) == 1              # exactly one foot on the ground
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)                   # don't reward standing on one leg forever
    cmd = env.command_manager.get_command(command_name)[:, [0, 2]]
    return reward * (torch.norm(cmd, dim=1) > 0.1)                # only when told to move


# ============================================================ foot / contact (needs contact sensor)
def foot_clearance(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``foot_clr``: fraction of legs whose measured swing (frc<1) agrees with the swing mask."""
    term = env.action_manager.get_term(action_name)
    swing_meas = _foot_force(env, sensor_cfg) < 1.0
    agree = torch.logical_and(swing_meas, term.swing_mask).float().sum(dim=1) / 2.0
    return agree * _static_flag(env, command_name)


def foot_support(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
                 sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``foot_supt``: fraction of legs whose measured support (frc>=10) agrees with support mask."""
    term = env.action_manager.get_term(action_name)
    support_meas = _foot_force(env, sensor_cfg) >= 10.0
    agree = torch.logical_and(support_meas, term.support_mask).float().sum(dim=1) / 2.0
    return agree * _static_flag(env, command_name)


def foot_height(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
                sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``foot_heit``: reward swing-foot clearance toward ~5 cm, penalise lifting too high / stance-foot lift.

    4 components: swing-mask clearance reward, over-height penalty, phase-support-mask lift penalty,
    and MEASURED-contact (frc>=10) lift penalty.
    """
    term = env.action_manager.get_term(action_name)
    fz = _foot_z(env, asset_cfg, action_name)                              # (N,2)
    sf = _static_flag(env, command_name)
    score = 40.0 * torch.clip(fz, 0.0, 0.05)
    rew = torch.clip(torch.sum(term.swing_mask.float() * score, dim=1), max=2.0) * sf
    rew = rew - 20.0 * torch.sum(torch.clip(fz - 0.06, min=0.0), dim=1)
    rew = rew - 0.2 * torch.sum(term.support_mask.float() * score, dim=1) * sf
    support_meas = (_foot_force(env, sensor_cfg) >= 10.0).float()          # measured contact
    rew = rew - 0.2 * torch.sum(support_meas * score, dim=1) * sf
    return rew


class foot_soft(ManagerTermBase):
    """RoboTamer ``foot_sft``: ``-0.1*clip(1/lvxn,0,1.5)*||frc_t - frc_{t-1}||/100`` (soft landing).

    Class-based term (keeps the per-env previous foot force, RoboTamer ``last_foot_frc``), so the
    penalty is on the force CHANGE per control step, not a constant contact-force tax on standing.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._last_frc = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids=None) -> None:
        self._last_frc[env_ids if env_ids is not None else slice(None)] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                 sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        frc = _foot_force(env, sensor_cfg)
        rew = (-0.1 * torch.clip(1.0 / _lin_vel_x_norm(env, command_name), 0.0, 1.5)
               * torch.norm(frc - self._last_frc, dim=1) / 100.0)
        self._last_frc.copy_(frc)
        return rew * _balance(env, command_name, asset_cfg)  # RoboTamer foot_sft = ... * balance_rew


def feet_contact_frc(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
                     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``feet_frc``: penalise swing-foot force; pull support force toward ~55 N."""
    term = env.action_manager.get_term(action_name)
    frc = _foot_force(env, sensor_cfg)
    sf = _static_flag(env, command_name)
    support_meas = frc >= 10.0
    swing_pen = -torch.norm(frc * term.swing_mask.float(), dim=1) * sf
    support_pen = -torch.norm(torch.clip((frc - 55.0).abs() * support_meas.float(), min=0.0), dim=1)
    return swing_pen + support_pen


def foot_acc(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``foot_acc``: ``-0.4*clip(1/lvxn,0,2)*||foot_vz(L,R)||`` (penalise vertical foot speed)."""
    lvxn = _lin_vel_x_norm(env, command_name)
    fvz = _foot_vel(env, asset_cfg, action_name)[:, :, 2]                 # (N,2)
    return -0.4 * torch.clip(1.0 / lvxn, 0.0, 2.0) * torch.norm(fvz, dim=1) * _balance(env)


def foot_py(env: ManagerBasedRLEnv, action_name: str = "birl_action",
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``feet_py``: ``-0.5*||foot_pitch(L,R)||`` (keep feet flat)."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    quat = asset.data.body_quat_w.torch[:, term.foot_body_ids, :]  # (N,2,4)
    _, pitch, _ = euler_xyz_from_quat(quat.reshape(-1, 4))
    pitch = wrap_to_pi(pitch).reshape(-1, 2)
    return -0.5 * torch.norm(pitch, dim=1) * _balance(env)


# ============================================================ smoothness (2nd-difference / jerk)
def action_smooth(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``act_smo``: ``-0.3*clip(1/lvxn,0,2)*||q_{t-2}-2q_{t-1}+q_t||`` (joint-target jerk)."""
    term = env.action_manager.get_term(action_name)
    lvxn = _lin_vel_x_norm(env, command_name)
    jerk = term.joint_act_prev2 - 2.0 * term.joint_act_prev + term.current_joint_act
    return -0.3 * torch.clip(1.0 / lvxn, 0.0, 2.0) * torch.norm(jerk, dim=1) * _balance(env)


def net_out_smooth(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``net_smo``: ``-0.2*clip(1/lvxn,0,2)*||(net 2nd-diff)[joints]||^2`` (rate-channel jerk)."""
    term = env.action_manager.get_term(action_name)
    lvxn = _lin_vel_x_norm(env, command_name)
    d2 = (term.net_out_prev2 - 2.0 * term.net_out_prev + term.net_out)[:, 2:]
    return -0.2 * torch.clip(1.0 / lvxn, 0.0, 2.0) * torch.sum(d2 * d2, dim=1) * _balance(env)


def pmf_smooth(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``pmf``: frequency-channel jerk + support-leg frequency penalty, static-gated."""
    term = env.action_manager.get_term(action_name)
    lvxn = _lin_vel_x_norm(env, command_name)
    sf = _static_flag(env, command_name)
    d2_freq = (term.net_out_prev2 - 2.0 * term.net_out_prev + term.net_out)[:, :2]
    rew = -0.02 * torch.clip(1.0 / lvxn, 0.0, 1.0) * torch.norm(d2_freq, dim=1)
    rew = rew - 0.5 * torch.clip(1.0 / lvxn, 0.0, 1.0) * torch.sum(
        torch.square(term.net_out[:, :2] * term.support_mask.float()), dim=1
    )
    return rew * sf * _balance(env)


def net_out_val(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``net_out_val``: ``-0.4*clip(1/lvxn,0,1)*||net_rate||^2`` (discourage large joint-rate output)."""
    term = env.action_manager.get_term(action_name)
    lvxn = _lin_vel_x_norm(env, command_name)
    return -0.4 * torch.clip(1.0 / lvxn, 0.0, 1.0) * torch.sum(torch.square(term.net_out[:, 2:]), dim=1) * _balance(env)


# ============================================================ posture / effort
def action_constraint(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``act_const``: pull joint target to ref; strong pull on hip yaw/roll while standing."""
    term = env.action_manager.get_term(action_name)
    lvxn = _lin_vel_x_norm(env, command_name)
    sf = _static_flag(env, command_name)
    dev = term.current_joint_act - term.ref_joint_pos
    rew = -0.1 * torch.clip(1.0 / lvxn, 0.0, 1.0) * torch.norm(dev, dim=1)
    rew = rew - 3.0 * torch.norm(dev[:, [0, 1, 5, 6]], dim=1) * sf       # hip_yaw/roll L,R
    return rew * _balance(env)


def joint_pos_error(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``jnt_pos_err``: ``-0.4*clip(1/lvxn,0,1)*||target - q||^2`` (PD tracking error)."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    q = asset.data.joint_pos.torch[:, term.joint_ids]
    lvxn = _lin_vel_x_norm(env, command_name)
    err = term.current_joint_act - q
    return -0.4 * torch.clip(1.0 / lvxn, 0.0, 1.0) * torch.sum(err * err, dim=1) * _balance(env)


def joint_vel(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``jnt_vel``: ``-0.4*clip(1/lvxn,0,1)*||qd||^2 - clip(1/lvxn,0,1)*||qd[hip_yaw,roll]||^2``."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    qd = asset.data.joint_vel.torch[:, term.joint_ids]
    c = torch.clip(1.0 / _lin_vel_x_norm(env, command_name), 0.0, 1.0)
    return (-0.4 * c * torch.sum(qd * qd, dim=1) - c * torch.sum(torch.square(qd[:, [0, 1, 5, 6]]), dim=1)) * _balance(env)


def joint_torque(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``joint_tor``: over-limit hinge ``-0.4*clip(1/lvxn,0,2)*sum((|tau|-limit)_+)*static``."""
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    tau = asset.data.applied_torque.torch[:, term.joint_ids].abs()
    lim = torch.tensor(_TORQUE_LIMIT, device=env.device)
    over = torch.clip(tau - lim, min=0.0).sum(dim=1)
    c = torch.clip(1.0 / _lin_vel_x_norm(env, command_name), 0.0, 2.0)
    return -0.4 * c * over * _static_flag(env, command_name)


def leg_width(env: ManagerBasedRLEnv, action_name: str = "birl_action",
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``leg_width_rew``: ``-||(|foot_y(L,R) - base_y| - 0.14)||`` (nominal half-stance ~0.14 m).

    NOTE: world-frame y (heading≈0 for forward walking); a heading-frame transform would be exact.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    foot_y = asset.data.body_pos_w.torch[:, term.foot_body_ids, 1]          # (N,2)
    base_y = asset.data.root_pos_w.torch[:, 1:2]                            # (N,1)
    return -torch.norm((foot_y - base_y).abs() - 0.14, dim=1) * _balance(env)


# ============================================================ style terms (balance-gated)
# Small-weight, static-gated style terms that refine an already-walking gait.
def base_acc(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
             action_name: str = "birl_action",
             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``base_acc``: penalise IMU proper acceleration minus g, static-gated.

    Tamer's ``env.base_acc`` is an IMU model (finite-diff of base lin vel, base frame, clip ±30,
    z += 9.8); the reward subtracts [0,0,9.81] in the same base-frame components. The action term's
    sensor buffers compute exactly that (un-biased ``last_raw_acc``).
    """
    term = env.action_manager.get_term(action_name)
    g = torch.tensor([0.0, 0.0, 9.81], device=env.device)
    lvxn = _lin_vel_x_norm(env, command_name)
    return -0.4 / lvxn * torch.norm((term.sensors.last_raw_acc - g) * 0.1, dim=1) * _static_flag(env, command_name) * _balance(env)


def sa_const(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action") -> torch.Tensor:
    """RoboTamer ``sa_const``: pull joint target to ref + support-leg deviation penalty, static-gated."""
    term = env.action_manager.get_term(action_name)
    c = torch.clip(1.0 / _lin_vel_x_norm(env, command_name), 0.0, 1.0)
    sf = _static_flag(env, command_name)
    dev = term.current_joint_act - term.ref_joint_pos                    # (N,10) deploy order L,R
    supL = term.support_mask[:, 0:1].float(); supR = term.support_mask[:, 1:2].float()
    pull = -0.1 * c * torch.sum(dev * dev, dim=1) * sf
    supdev = torch.sum(torch.square(dev[:, :5]) * supL, dim=1) + torch.sum(torch.square(dev[:, 5:]) * supR, dim=1)
    return (pull - c * supdev * sf) * _balance(env)


def foot_slip(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``foot_slip`` — 4 parts:
    1. ``+2*(lvxn*Σ(foot_vx*sign(cmd_x)*swing)).clip(0,1)*static`` — reward SWING foot advancing in the
       command direction.
    2. ``-0.5*||foot_vy(L,R)||*static`` — penalise lateral foot velocity while walking.
    3. ``+0.3*||foot_v_xy(L,R)||*(static-1)`` — while STANDING, small pull toward foot motion.
    4. ``-0.3/lvxn*|| 0.1*foot_v_xy/clip_foot_h * support ||*static`` — penalise STANCE-foot sliding.
    ``clip_foot_h = |foot_z| + 0.03``.
    """
    term = env.action_manager.get_term(action_name)
    fv = _foot_vel(env, asset_cfg, action_name)                          # (N,2,3) world (L,R)
    lvxn = _lin_vel_x_norm(env, command_name)
    sf = _static_flag(env, command_name)
    swing = term.swing_mask.float()
    support = term.support_mask.float()
    cmd_x_sign = torch.sign(env.command_manager.get_command(command_name)[:, 0])   # (N,)
    clip_foot_h = _foot_z(env, asset_cfg, action_name).abs() + 0.03      # (N,2)

    # part 1: swing foot advancing along command direction
    advance = torch.sum(fv[:, :, 0] * cmd_x_sign.unsqueeze(1) * swing, dim=1)
    rew = 2.0 * torch.clip(lvxn * advance, 0.0, 1.0) * sf
    # part 2: lateral (y) foot velocity while walking
    rew = rew - 0.5 * torch.norm(fv[:, :, 1], dim=1) * sf
    # part 3: while standing (static-1) reverse branch
    rew = rew + 0.3 * torch.norm(torch.norm(fv[:, :, :2], dim=-1), dim=1) * (sf - 1.0)
    # part 4: stance-foot slide normalised by foot height
    slide = torch.norm(0.1 * torch.norm(fv[:, :, :2], dim=-1) / clip_foot_h * support, dim=1)
    rew = rew - 0.3 / lvxn * slide * sf
    return rew * _balance(env)


def foot_vz(env: ManagerBasedRLEnv, command_name: str = "base_velocity", action_name: str = "birl_action",
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """RoboTamer ``foot_vz`` — penalise DOWNWARD foot velocity (soft landing):
    1. ``-0.1*clip(1/lvxn,0,1)*||foot_vz.clip(max=0)/clip_foot_h||*static``  (while walking)
    2. ``+0.8*clip(1/lvxn,0,1)*||foot_vz.clip(max=0)||*(static-1)``          (while standing)
    ``clip_foot_h = |foot_z| + 0.03``. NOTE: applied to BOTH feet; the ``/clip_foot_h`` normaliser makes
    it fire mostly on the airborne (swing) foot.
    """
    fvz = _foot_vel(env, asset_cfg, action_name)[:, :, 2]                # (N,2)
    down = torch.clip(fvz, max=0.0)                                      # downward only (<=0)
    clip_foot_h = _foot_z(env, asset_cfg, action_name).abs() + 0.03      # (N,2)
    c = torch.clip(1.0 / _lin_vel_x_norm(env, command_name), 0.0, 1.0)
    sf = _static_flag(env, command_name)
    rew = -0.1 * c * torch.norm(down / clip_foot_h, dim=1) * sf          # part 1 (while walking)
    rew = rew + 0.8 * c * torch.norm(down, dim=1) * (sf - 1.0)           # part 2 (while standing)
    return rew * _balance(env)


# ============================================================ balance gate (RoboTamer coupling)
def balance_factor(env: ManagerBasedRLEnv, command_name: str = "base_velocity",
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Public accessor for RoboTamer's ``balance ∈ [0.5,1]`` gate — used by the standalone ``balance``
    RewTerm (= RoboTamer ``balance_rew*1.5``)."""
    return _balance(env, command_name, asset_cfg)


# RoboTamer multiplies these style/foot/smooth terms by balance_rew — each applies ``* _balance(env)``
# inline in its return: foot_phase_antiphase, foot_acc, foot_py, action_smooth, net_out_smooth,
# pmf_smooth, net_out_val, action_constraint, joint_pos_error, joint_vel, leg_width, base_acc, sa_const,
# foot_slip, foot_vz. foot_soft (class) gates itself in __call__. NOT gated: the tracking core,
# base_height, alive, foot_clr/supt/heit, feet_frc, joint_tor.
