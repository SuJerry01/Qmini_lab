# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Custom observation terms for the Qmini BIRL gait (RoboTamer's obs).

The policy obs is a SINGLE 43-dim term (:func:`policy_obs`) so the manager's 3-frame stack
(history_length=3, flatten_history_dim=True) yields the deploy layout [frame0(43),frame1,frame2]=129
the C++ ONNX SDK expects. The [-3,3] clip is set on the ObsTerm (RoboTamer .clip(-3,3)). Per-block
scales (x0.5 ang-vel, x0.1 joint-vel, x1.0 euler) are baked in; gait blocks are static-gated.
:func:`critic_privileged` adds the training-only asymmetric-critic ground truth.

Articulation is imported under TYPE_CHECKING only (runtime import pulls in pxr; this module is
imported during hydra cfg resolution BEFORE SimulationApp).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


# ----------------------------------------------------------------------------- helpers
def _static_flag(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """``1`` iff commanded (``||[vx, yaw]|| >= threshold``), else ``0`` — shape ``(N,1)``."""
    cmd = env.command_manager.get_command(command_name)[:, [0, 2]]
    return (torch.norm(cmd, dim=1, keepdim=True) >= threshold).float()


# ----------------------------------------------------------------------------- policy obs (43, single term)
def policy_obs(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    action_name: str = "birl_action",
    threshold: float = 0.15,
    ang_vel_scale: float = 0.5,
    joint_vel_scale: float = 0.1,
    sensor_model: bool = True,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The full 43-dim RoboTamer pure observation, in deploy order (unclipped; ObsTerm sets clip).

    Layout (43) = cmd[vx,yaw](2) + euler[roll,pitch]x1(2) + ang_vel x0.5(3) + (q-ref)(10) +
    q_dot x0.1(10) + joint_pos_error(10) + phase[sinL,sinR,cosL,cosR]*static(4) + (f*0.3-1)*static(2).

    ``sensor_model`` reads proprioception through the RoboTamer sensor chain (fixed per-env bias +
    1 kHz ring buffers + global random read delay); joint_pos_error uses the SAME delayed+biased q as
    (q-ref) so the two slices stay anticorrelated. sensor_model=False = clean fresh reads (debug only).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)

    cmd = env.command_manager.get_command(command_name)[:, [0, 2]]                 # 2
    if sensor_model:
        buf = term.sensors
        euler = buf.delayed_base_euler()[:, :2]                                    # 2 (delayed + biased)
        ang_vel_raw = buf.delayed_base_ang_vel()                                   # 3
        q = buf.delayed_joint_pos()                                                # 10
        qd = buf.delayed_joint_vel()                                               # 10
    else:
        jids = term.joint_ids
        roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
        euler = torch.stack([wrap_to_pi(roll), wrap_to_pi(pitch)], dim=-1)
        ang_vel_raw = asset.data.root_ang_vel_b.torch
        q = asset.data.joint_pos.torch[:, jids]
        qd = asset.data.joint_vel.torch[:, jids]

    ang_vel = ang_vel_raw * ang_vel_scale                                         # 3
    q_minus_ref = q - term.ref_joint_pos                                          # 10
    qd_scaled = qd * joint_vel_scale                                              # 10
    joint_pos_error = term.current_joint_act - q                                 # 10
    sf = _static_flag(env, command_name, threshold)                              # (N,1)
    phase = term.phase_sin_cos * sf                                              # 4
    freq = term.freq_feature * sf                                                # 2
    return torch.cat([cmd, euler, ang_vel, q_minus_ref, qd_scaled, joint_pos_error, phase, freq], dim=-1)


# ----------------------------------------------------------------------------- critic privileged (training-only)
def critic_privileged(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["ankle_pitch_.*"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "birl_action",
) -> torch.Tensor:
    """Privileged FRESH-state frame (64) — Tamer's asymmetric critic ground truth.

    The delayed actor 43 already sits in the critic group via PolicyCfg inheritance; this term carries
    the fresh/ground-truth remainder, with Tamer's clips/scales:

    cmd errors(2) + base_lin_vel_b(3) + fresh euler rp(2) + fresh ang_vel x0.5(3) + fresh (q-ref)(10) +
    fresh qd x0.1(10) + (joint_act-ref)(10) + net_out[2:]/15(10) + foot_z clip(±0.5)x10(2) +
    (base_z-0.4)x10(1) + foot_vel clip(±8)x0.5(6) + IMU base_acc clip(±20)x0.2(3) +
    foot_frc clip(0,200)x0.01(2) = 64. Not deployed — layout need not match any contract.
    NOTE: foot_vel is world-frame (Tamer: heading frame) — same approximation as rewards.leg_width.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    jids = term.joint_ids
    cmd = env.command_manager.get_command(command_name)                             # (N,3): vx, vy, wz
    lin_vel_b = asset.data.root_lin_vel_b.torch                              # 3
    ang_vel_b = asset.data.root_ang_vel_b.torch
    cmd_err = torch.stack([cmd[:, 0] - lin_vel_b[:, 0], cmd[:, 2] - ang_vel_b[:, 2]], dim=-1)  # 2
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    euler = torch.stack([wrap_to_pi(roll), wrap_to_pi(pitch)], dim=-1)              # 2 (fresh)
    q = asset.data.joint_pos.torch[:, jids]                                  # (N,10) fresh
    qd = asset.data.joint_vel.torch[:, jids]
    base_z = asset.data.root_pos_w.torch[:, 2:3]                             # 1
    foot_z = asset.data.body_pos_w.torch[:, term.foot_body_ids, 2] - 0.1     # 2 (sole clearance)
    foot_vel = asset.data.body_lin_vel_w.torch[:, term.foot_body_ids, :].reshape(env.num_envs, 6)
    if "contact_forces" in env.scene.sensors:
        sensor = env.scene.sensors["contact_forces"]
        foot_force = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :], dim=-1)  # 2
    else:
        foot_force = torch.zeros_like(foot_z)
    return torch.cat(
        [
            cmd_err,                                                                # 2
            lin_vel_b,                                                              # 3
            euler,                                                                  # 2
            ang_vel_b * 0.5,                                                        # 3
            q - term.ref_joint_pos,                                                 # 10
            qd * 0.1,                                                               # 10
            term.current_joint_act - term.ref_joint_pos,                            # 10
            term.net_out[:, 2:] / 15.0,                                             # 10
            foot_z.clip(-0.5, 0.5) * 10.0,                                          # 2
            (base_z - 0.4) * 10.0,                                                  # 1
            foot_vel.clip(-8.0, 8.0) * 0.5,                                         # 6
            term.sensors.last_raw_acc.clip(-20.0, 20.0) * 0.2,                      # 3 (IMU, un-biased)
            foot_force.clip(0.0, 200.0) * 0.01,                                     # 2
        ],
        dim=-1,
    )


# ----------------------------------------------------------------------------- per-block helpers (reference)
def commands_vx_yaw(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, [0, 2]]


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_lin_vel_b.torch
