# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Custom stateful ActionTerm for the Qmini BIRL gait (RoboTamer's action).

12-dim action = 2 phase-frequencies + 10 incremental joint targets. The term owns the gait state:
a PhaseModulator, the running incremental joint target, and the short histories the smoothness
rewards read (RoboTamer net_out_history / action_history). process_actions de-scales/integrates once
per env step; apply_actions writes the held target each substep; reset restores the ref pose. Obs and
reward terms read the state via env.action_manager.get_term("birl_action").
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from qmini_tasks.assets.qmini import (
    QMINI_FOOT_BODIES,
    QMINI_JOINT_ORDER,
    QMINI_POS_LIMIT_HIGH,
    QMINI_POS_LIMIT_LOW,
    QMINI_REF_JOINT_POS,
)
from qmini_tasks.tasks.direct.qmini_birl.phase_modulator import PhaseModulator

from .sensor_delay import SensorDelayBuffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class QminiBirlAction(ActionTerm):
    """12-dim BIRL action: ``[2 phase-frequency, 10 incremental joint targets]``."""

    cfg: QminiBirlActionCfg

    def __init__(self, cfg: QminiBirlActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        dev = self.device
        # joints resolved in the deploy contract order
        self._joint_ids, self._joint_names = self._asset.find_joints(list(QMINI_JOINT_ORDER), preserve_order=True)
        self._num_joints = len(self._joint_ids)
        if self._num_joints != 10:
            raise ValueError(f"Qmini BIRL action expects 10 joints, resolved {self._num_joints}: {self._joint_names}")
        # foot bodies in deploy order (L, R) — matches the per-leg support/swing masks.
        self._foot_body_ids, self._foot_body_names = self._asset.find_bodies(
            list(QMINI_FOOT_BODIES), preserve_order=True
        )
        # constants
        self._ref = torch.tensor(QMINI_REF_JOINT_POS, device=dev).repeat(self.num_envs, 1)  # (N,10)
        # PD feedforward torque bias (RoboTamer joint_tor_offset + a constant "+d_gains" term), deploy
        # order. Added as effort on top of the implicit PD (does not change the position target);
        # hip_roll also gets Coulomb friction in apply_actions. Both are velocity-independent constants.
        _joint_bias = [0.6, 1.0, 0.0, 0.7, 0.0, -0.6, -1.0, -0.0, -0.7, 0.0]     # joint_tor_offset
        _d_gains_const = [0.3, 2.5, 0.3, 0.5, 0.25, 0.3, 2.5, 0.3, 0.5, 0.25]    # "+ d_gains"
        self._tor_offset = (
            torch.tensor(_joint_bias, device=dev) + torch.tensor(_d_gains_const, device=dev)
        )  # (10,)
        self._vel_sign_mask = torch.tensor(
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=dev
        )  # hip_roll only (10,)
        self._phase_low = torch.full((2,), cfg.phase_freq_range[0], device=dev)
        self._phase_high = torch.full((2,), cfg.phase_freq_range[1], device=dev)
        self._joint_low = torch.full((self._num_joints,), cfg.joint_rate_range[0], device=dev)
        self._joint_high = torch.full((self._num_joints,), cfg.joint_rate_range[1], device=dev)
        # Joint position limits = the deploy contract (act_pos_low/high), NOT the USD's (importer
        # sign-flipped knee/hip_pitch). Clip the target to these + write to sim on first reset so PhysX
        # doesn't clamp the knee straight.
        self._lim_low = torch.tensor(QMINI_POS_LIMIT_LOW, device=dev).repeat(self.num_envs, 1)   # (N,10)
        self._lim_high = torch.tensor(QMINI_POS_LIMIT_HIGH, device=dev).repeat(self.num_envs, 1)
        self._limits_written = False
        # gait clock (shared component)
        self._pm = PhaseModulator(
            self.num_envs, dev, num_legs=2, rest_frequency=cfg.rest_frequency, convert_phi=cfg.convert_phi
        )
        # buffers
        self._raw = torch.zeros(self.num_envs, 12, device=dev)
        self._processed = torch.zeros(self.num_envs, 12, device=dev)
        self._joint_act = self._ref.clone()          # running incremental joint target (deploy order)
        self._freq = torch.full((self.num_envs, 2), cfg.rest_frequency, device=dev)
        # short histories (t, t-1, t-2) for the 2nd-difference smoothness rewards
        self._net_out = torch.zeros(self.num_envs, 12, device=dev)        # de-scaled [freq(2), rate(10)]
        self._net_out_prev = torch.zeros_like(self._net_out)
        self._net_out_prev2 = torch.zeros_like(self._net_out)
        self._joint_act_prev = self._ref.clone()
        self._joint_act_prev2 = self._ref.clone()
        # RoboTamer sensor model: fixed per-env bias + 1 kHz ring buffers + global random read delay.
        # Appended each substep in apply_actions; obs terms read the delayed views via .sensors.
        self._sensors = SensorDelayBuffers(
            self.num_envs,
            dev,
            sim_dt=self._env.physics_dt,
            delay_enabled=cfg.sensor_delay,
            noise_enabled=cfg.sensor_bias_noise,
            delay_joint_range=cfg.delay_joint_range,
            delay_rate_range=cfg.delay_rate_range,
            delay_angle_range=cfg.delay_angle_range,
            resample_every_steps=cfg.delay_resample_every,
        )
        # Per-env torque-scale DR (Tamer tau_gains): drawn by the randomize_pd_torque_gains reset event;
        # scales the feedforward effort here. Ones until the first reset.
        self._tau_scale = torch.ones(self.num_envs, self._num_joints, device=dev)

    # --- required ActionTerm interface ---
    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    # --- public accessors for observation / reward functions ---
    @property
    def joint_ids(self) -> Sequence[int]:
        return self._joint_ids

    @property
    def foot_body_ids(self) -> Sequence[int]:
        return self._foot_body_ids

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self._ref

    @property
    def current_joint_act(self) -> torch.Tensor:
        return self._joint_act

    @property
    def joint_act_prev(self) -> torch.Tensor:
        return self._joint_act_prev

    @property
    def joint_act_prev2(self) -> torch.Tensor:
        return self._joint_act_prev2

    @property
    def net_out(self) -> torch.Tensor:
        """De-scaled network output ``[freq(2), rate(10)]`` at t (RoboTamer ``net_out_history[-1]``)."""
        return self._net_out

    @property
    def net_out_prev(self) -> torch.Tensor:
        return self._net_out_prev

    @property
    def net_out_prev2(self) -> torch.Tensor:
        return self._net_out_prev2

    @property
    def phase_sin_cos(self) -> torch.Tensor:
        return self._pm.sin_cos          # (N,4)

    @property
    def freq_feature(self) -> torch.Tensor:
        return self._pm.freq_feature     # (N,2) = f*0.3 - 1

    @property
    def frequency(self) -> torch.Tensor:
        return self._freq                # (N,2) de-scaled leg frequencies

    @property
    def support_mask(self) -> torch.Tensor:
        return self._pm.support_mask     # (N,2) bool

    @property
    def swing_mask(self) -> torch.Tensor:
        return self._pm.swing_mask       # (N,2) bool

    @property
    def sensors(self) -> SensorDelayBuffers:
        """Biased + delayed sensor views (RoboTamer sensor model) — read by the obs terms."""
        return self._sensors

    @property
    def tau_scale(self) -> torch.Tensor:
        """Per-env torque-scale DR (N,10); written by the ``randomize_pd_torque_gains`` event."""
        return self._tau_scale

    @staticmethod
    def _scale(a: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        return (a + 1.0) * 0.5 * (hi - lo) + lo

    # --- operations ---
    def process_actions(self, actions: torch.Tensor) -> None:
        # global delay re-draw every 200 control steps (Tamer birl_task.py:192-198)
        self._sensors.maybe_resample(self._env.common_step_counter)
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        self._processed[:] = a
        self._freq = self._scale(a[:, :2], self._phase_low, self._phase_high)        # 2 leg frequencies
        rate = self._scale(a[:, 2:], self._joint_low, self._joint_high)              # 10 joint rates
        # roll net_out history (de-scaled 12-vec) — newest at t
        self._net_out_prev2 = self._net_out_prev.clone()
        self._net_out_prev = self._net_out.clone()
        self._net_out = torch.cat([self._freq, rate], dim=1)
        # advance the oscillator once per policy step
        self._pm.set_frequency(self._freq)
        self._pm.step(self._env.step_dt)
        # roll joint-target history, then integrate the new incremental target
        self._joint_act_prev2 = self._joint_act_prev.clone()
        self._joint_act_prev = self._joint_act.clone()
        self._joint_act = torch.clip(self._joint_act + rate * self._env.step_dt, self._lim_low, self._lim_high)

    def apply_actions(self) -> None:
        # held constant across the decimation PD loop
        self._asset.set_joint_position_target_index(target=self._joint_act, joint_ids=self._joint_ids)
        # PD feedforward: constant offset + hip_roll Coulomb friction -3.5*sign(qd) (recomputed each
        # substep from live joint vel), scaled by tau_scale (Tamer tau_gains torque DR).
        qd = self._asset.data.joint_vel.torch[:, self._joint_ids]                 # (N,10) deploy order
        ff = (self._tor_offset - 3.5 * torch.sign(qd) * self._vel_sign_mask) * self._tau_scale
        self._asset.set_joint_effort_target_index(target=ff, joint_ids=self._joint_ids)
        # sensor model tick (1 kHz): append one substep of raw state to the delay ring buffers.
        self._sensors.append(
            joint_pos=self._asset.data.joint_pos.torch[:, self._joint_ids],
            joint_vel=qd,
            root_quat_w=self._asset.data.root_quat_w.torch,
            root_ang_vel_b=self._asset.data.root_ang_vel_b.torch,
            root_lin_vel_w=self._asset.data.root_lin_vel_w.torch,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # once, after sim is live: overwrite the (sign-flipped USD) joint position limits with the
        # deploy-contract limits so PhysX no longer clamps the knee/hip straight.
        if not self._limits_written:
            limits = torch.stack([self._lim_low[0], self._lim_high[0]], dim=-1)      # (10,2)
            limits = limits.unsqueeze(0).repeat(self.num_envs, 1, 1)                 # (N,10,2)
            self._asset.write_joint_position_limit_to_sim(limits, joint_ids=self._joint_ids)
            self._limits_written = True
        if env_ids is None:
            buf_ids = slice(None)
            pm_ids = torch.arange(self.num_envs, device=self.device)
        else:
            buf_ids = env_ids
            pm_ids = torch.as_tensor(env_ids, device=self.device).long().flatten()
        self._raw[buf_ids] = 0.0
        self._processed[buf_ids] = 0.0
        self._joint_act[buf_ids] = self._ref[buf_ids]
        self._joint_act_prev[buf_ids] = self._ref[buf_ids]
        self._joint_act_prev2[buf_ids] = self._ref[buf_ids]
        self._freq[buf_ids] = self.cfg.rest_frequency
        self._net_out[buf_ids] = 0.0
        self._net_out_prev[buf_ids] = 0.0
        self._net_out_prev2[buf_ids] = 0.0
        self._pm.reset(pm_ids, randomize=True)
        # flood the sensor rings with the POST-teleport state (reset events run before this).
        # NOTE: _tau_scale is NOT reset here — the randomize_pd_torque_gains event owns that buffer.
        self._sensors.reset(
            pm_ids,
            joint_pos=self._asset.data.joint_pos.torch[pm_ids][:, self._joint_ids],
            joint_vel=self._asset.data.joint_vel.torch[pm_ids][:, self._joint_ids],
            root_quat_w=self._asset.data.root_quat_w.torch[pm_ids],
            root_ang_vel_b=self._asset.data.root_ang_vel_b.torch[pm_ids],
            root_lin_vel_w=self._asset.data.root_lin_vel_w.torch[pm_ids],
        )


@configclass
class QminiBirlActionCfg(ActionTermCfg):
    """Config for :class:`QminiBirlAction`."""

    class_type: type = QminiBirlAction
    asset_name: str = "robot"
    phase_freq_range: tuple[float, float] = (0.5, 3.5)      # inc_high/low_ranges[:2]
    joint_rate_range: tuple[float, float] = (-15.0, 15.0)   # inc_high/low_ranges[2:]
    rest_frequency: float = 0.5
    convert_phi: float = 1.2 * math.pi
    # RoboTamer sensor model: global random obs delay over 1 kHz buffers + fixed per-env bias.
    # Ranges in sim substeps (= ms at dt 0.001).
    sensor_delay: bool = True                               # delay_observation
    sensor_bias_noise: bool = True                          # bias, not white noise
    delay_joint_range: tuple[int, int] = (10, 40)           # delay_joint_ranges
    delay_rate_range: tuple[int, int] = (20, 50)            # delay_rate_ranges
    delay_angle_range: tuple[int, int] = (20, 50)           # delay_angle_ranges
    delay_resample_every: int = 200                         # control steps
