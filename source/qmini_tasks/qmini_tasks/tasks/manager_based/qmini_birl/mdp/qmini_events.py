# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Tamer-faithful DR events: periodic push + PD/torque-gain DR + correlated mass/inertia DR.

RoboTamer's periodic push is NOT an additive kick: every 3 s it (a) REPLACES the root velocity with
bounded U(-0.5r, 0.5r), (b) SNAPS the base orientation to a fresh near-upright tilt (rpy ~ U(-0.2,
0.2)) — a rescue that keeps episodes alive for gait rewards — and (c) RAMPS r 1.0->1.5 over ~3000
iterations. The stock push_by_setting_velocity ADDS velocity, never rights the base, has no ramp.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _u(lo_hi: tuple[float, float], shape: tuple[int, ...], device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (lo_hi[1] - lo_hi[0]) + lo_hi[0]


# Nominal SIM PD stiffness in deploy joint order (= qmini.py actuators / Tamer pd_gains.stiffness);
# nominal velocity damping is 1.0 uniform (see qmini.py's damping note).
_NOMINAL_KP = [55.0, 105.0, 75.0, 45.0, 30.0, 55.0, 105.0, 75.0, 45.0, 30.0]


def randomize_pd_torque_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gains_range: tuple[float, float] = (0.8, 1.2),
    torque_range: tuple[float, float] = (0.8, 1.2),
    action_name: str = "birl_action",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Tamer's PD-gain + torque-scale DR, re-drawn per RESET.

    Tamer's applied torque is ``tau_gains * clip(p_gains*p_rand*err + const - d_rand*qd + ff, ±lim)``
    with ``p_rand, d_rand, tau_gains ~ U(0.8,1.2)`` re-sampled per reset. With an implicit PD actuator
    that factors EXACTLY into: stiffness = kp*p_rand*tau, damping = 1.0*d_rand*tau, and the feedforward
    effort x tau (the action term reads ``tau_scale``). The inner clip-before-tau is not portable but
    only binds when PD saturates (mid-fall).
    """
    asset = env.scene[asset_cfg.name]
    term = env.action_manager.get_term(action_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    p_rand = _u(gains_range, (n, 10), env.device)
    d_rand = _u(gains_range, (n, 10), env.device)
    tau = _u(torque_range, (n, 10), env.device)
    kp = torch.tensor(_NOMINAL_KP, device=env.device)
    asset.write_joint_stiffness_to_sim_index(stiffness=kp * p_rand * tau, joint_ids=term.joint_ids, env_ids=env_ids)
    asset.write_joint_damping_to_sim_index(damping=d_rand * tau, joint_ids=term.joint_ids, env_ids=env_ids)
    term.tau_scale[env_ids] = tau


def randomize_mass_inertia_tamer(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    mass_range: tuple[float, float] = (0.5, 1.5),
    inertia_range: tuple[float, float] = (0.5, 1.5),
    base_add_range: tuple[float, float] = (-0.6, 0.7),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base_link"),
):
    """Tamer's creation-time mass/inertia DR (recomputeInertia=False, so the manual inertia scale is live).

    Per env: ONE ``dm ~ U(0.5,1.5)`` scales every NON-base link mass (correlated across links); ONE
    ``di ~ U(0.5,1.5)`` scales every link's inertia diagonal; the base mass gets an ADDITIVE
    ``U(-0.6,+0.7) * m0`` payload. Startup-only.
    """
    asset = env.scene[asset_cfg.name]
    n = env.num_envs
    base_ids = torch.tensor(asset_cfg.body_ids, device=env.device)
    masses = asset.data.body_mass.torch.clone()            # (N, B)
    inertias = asset.data.body_inertia.torch.clone()       # (N, B, 9)
    dm = _u(mass_range, (n, 1), env.device)
    di = _u(inertia_range, (n, 1, 1), env.device)
    base_m0 = masses[:, base_ids].clone()
    masses *= dm                                                   # all links x dm ...
    masses[:, base_ids] = base_m0 + _u(base_add_range, (n, len(base_ids)), env.device) * base_m0  # ... base additive
    inertias *= di                                                 # every link's inertia x di
    asset.set_masses_index(masses=masses, full_data=True)
    asset.set_inertias_index(inertias=inertias, full_data=True)


def push_replace_vel_and_right(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_magnitude: float = 0.5,
    tilt: float = 0.2,
    ramp_end_iters: int = 3000,
    steps_per_iter: int = 24,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """REPLACE root velocity with U(-m, m) and snap orientation to a fresh ±tilt rpy; m ramps 1->1.5x."""
    asset = env.scene[asset_cfg.name]
    dev = env.device
    n = len(env_ids)
    # magnitude ramp 1.0 -> 1.5 over ramp_end_iters policy iterations
    frac = min(1.0, float(env.common_step_counter) / float(ramp_end_iters * steps_per_iter))
    m = base_magnitude * (1.0 + 0.5 * frac)
    # (b) rescue: fresh near-upright orientation, position unchanged
    rpy = (torch.rand(n, 3, device=dev) * 2.0 - 1.0) * tilt
    quat = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    pos = asset.data.root_pos_w.torch[env_ids]
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    # (a) velocity REPLACE (not +=), bounded
    vel = (torch.rand(n, 6, device=dev) * 2.0 - 1.0) * m
    asset.write_root_velocity_to_sim(vel, env_ids=env_ids)
