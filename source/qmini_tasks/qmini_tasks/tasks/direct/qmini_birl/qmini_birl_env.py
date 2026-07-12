# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Qmini BIRL locomotion task — Direct workflow (Isaac Lab reconstruction of RoboTamer4Qmini.BIRLTask).

This is a *reconstruction*, not a port: each piece is re-expressed in idiomatic Isaac Lab
``DirectRLEnv`` lifecycle methods, preserving the *purpose* of the source functions (see
``docs/deep_dive.md`` for the WHAT/WHY/HOW). Mapping:

- ``_pre_physics_step``  <- BIRLTask.action: de-scale net output, drive the PhaseModulator with the 2
  frequency channels, integrate the 10 incremental joint targets, advance the command timer.
- ``_apply_action``      -> hold the joint position target across the 15-step decimation PD loop.
- ``_get_observations``  <- BIRLTask.pure_observation (43, clipped, x3-stacked = 129) for "policy",
  plus a privileged superset for "critic" (asymmetric actor-critic).
- ``_get_rewards``       <- BIRLTask.reward (faithful subset, grouped by purpose; weighted sum * dt).
- ``_get_dones``         <- BaseTask.terminate (tilt / base height / undesired contact / timeout).
- ``_reset_idx``         <- env+task reset: ref pose, randomized phase, command resample, buffers.

Joint order is the sim-to-real contract: everything is kept in :data:`QMINI_JOINT_ORDER` via the
``self._joint_ids`` remap so the 43-d obs / 12-d action match the deployed ONNX exactly.

M1 scope / deferred (clearly-marked TODOs): per-term reward clip & the shared ``balance`` gate, the
full ~30-term reward set, domain randomization (events), and the randomized observation delay are
follow-ups (M2/M3). The structure, contract dims, gait clock, and action/obs pipeline are complete.
"""

from __future__ import annotations

import torch
import warp as wp
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from qmini_tasks.assets.qmini import (
    QMINI_FOOT_BODIES,
    QMINI_JOINT_ORDER,
    QMINI_REF_JOINT_POS,
    QMINI_TERMINATION_BODIES,
)

from .phase_modulator import PhaseModulator
from .qmini_birl_env_cfg import OBS_STACK, PURE_OBS_DIM, QminiLabEnvCfg


class QminiLabEnv(DirectRLEnv):
    cfg: QminiLabEnvCfg

    def __init__(self, cfg: QminiLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)  # builds the scene via _setup_scene()

        # --- joint/body index remaps to the deploy contract order ---
        self._joint_ids, _ = self.robot.find_joints(QMINI_JOINT_ORDER, preserve_order=True)
        self._foot_body_ids, _ = self.robot.find_bodies(QMINI_FOOT_BODIES, preserve_order=True)
        # ContactSensor uses find_sensors() in 3.0 (find_bodies is deprecated); returns (ids, names).
        # Contact-free mode (nested-URDF workaround): sensor is None -> ids unused (see cfg.use_contact_sensor).
        if self._contact_sensor is not None:
            self._foot_contact_ids, _ = self._contact_sensor.find_sensors(QMINI_FOOT_BODIES, preserve_order=True)
            self._term_contact_ids, _ = self._contact_sensor.find_sensors(QMINI_TERMINATION_BODIES)
        self._num_act_joints = len(self._joint_ids)  # 10

        # --- constant tensors ---
        self._ref_joint_pos = torch.tensor(QMINI_REF_JOINT_POS, device=self.device).repeat(self.num_envs, 1)
        # de-scale ranges (scale_transform), per channel
        self._phase_low = torch.full((2,), self.cfg.phase_freq_range[0], device=self.device)
        self._phase_high = torch.full((2,), self.cfg.phase_freq_range[1], device=self.device)
        self._joint_low = torch.full((self._num_act_joints,), self.cfg.joint_rate_range[0], device=self.device)
        self._joint_high = torch.full((self._num_act_joints,), self.cfg.joint_rate_range[1], device=self.device)
        # soft joint position limits (remapped to deploy order) for the incremental-target clip.
        # NOTE(3.0): robot.data.* are wp.array (Warp) in Isaac Lab 3.0 — wrap reads with wp.to_torch().
        soft_lim = wp.to_torch(self.robot.data.soft_joint_pos_limits)[:, self._joint_ids, :]  # (N, 10, 2)
        self._joint_limit_low = soft_lim[..., 0]
        self._joint_limit_high = soft_lim[..., 1]

        # --- gait clock ---
        self._phase = PhaseModulator(
            self.num_envs, self.device, num_legs=2,
            rest_frequency=self.cfg.phase_rest_frequency, convert_phi=self.cfg.phase_convert_phi,
        )

        # --- task buffers (all in deploy joint order) ---
        self._joint_target = self._ref_joint_pos.clone()      # current commanded joint target (incremental)
        self._prev_joint_target = self._ref_joint_pos.clone()
        self._prev2_joint_target = self._ref_joint_pos.clone()
        self._cur_freq = torch.full((self.num_envs, 2), self.cfg.phase_rest_frequency, device=self.device)
        self._prev_freq = self._cur_freq.clone()
        self._prev2_freq = self._cur_freq.clone()
        self._commands = torch.zeros(self.num_envs, 2, device=self.device)        # [vx, yaw]
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self._static_flag = torch.zeros(self.num_envs, 1, device=self.device)
        self._obs_history = torch.zeros(self.num_envs, OBS_STACK, PURE_OBS_DIM, device=self.device)
        self._died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # initialize commands/static cohort for all envs
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor_cfg) if self.cfg.use_contact_sensor else None
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        if self._contact_sensor is not None:
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scale(action: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """``scale_transform``: action in [-1,1] -> [low, high] via ``(a+1)/2*(high-low)+low``."""
        return (action + 1.0) * 0.5 * (high - low) + low

    def _compute_pure_obs(self) -> torch.Tensor:
        """Build the 43-dim pure observation in deploy order, clipped to [-3, 3]."""
        d = self.robot.data
        roll, pitch, _ = euler_xyz_from_quat(wp.to_torch(d.root_quat_w))
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        ang_vel = wp.to_torch(d.root_ang_vel_b)
        jp = wp.to_torch(d.joint_pos)[:, self._joint_ids]
        jv = wp.to_torch(d.joint_vel)[:, self._joint_ids]
        joint_pos_err = self._joint_target - jp  # PD tracking-error proxy (current_joint_act - q)
        pure = torch.cat(
            [
                self._commands,                                       # 2  [vx_cmd, yaw_cmd]
                torch.stack([roll, pitch], dim=-1),                   # 2  base euler roll/pitch (x1.0)
                ang_vel * self.cfg.ang_vel_scale,                     # 3  base ang vel x0.5
                jp - self._ref_joint_pos,                             # 10 (joint_pos - ref)
                jv * self.cfg.joint_vel_scale,                        # 10 joint_vel x0.1
                joint_pos_err,                                        # 10 joint_pos_error
                self._phase.sin_cos * self._static_flag,              # 4  phase sin/cos (gated)
                self._phase.freq_feature * self._static_flag,         # 2  (pm_f*0.3-1) (gated)
            ],
            dim=-1,
        )
        return pure.clip(-self.cfg.obs_clip, self.cfg.obs_clip)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        vx = torch.empty(n, device=self.device).uniform_(*self.cfg.command_vx_range)
        yaw = torch.empty(n, device=self.device).uniform_(*self.cfg.command_yaw_range)
        self._commands[env_ids, 0] = vx
        self._commands[env_ids, 1] = yaw
        self._command_time_left[env_ids] = self.cfg.command_resample_time_s
        # static cohort: force the first fraction of envs to zero command (keeps a standing regime)
        n_static = int(self.cfg.static_command_fraction * self.num_envs)
        if n_static > 0:
            self._commands[env_ids[env_ids < n_static]] = 0.0
        # recompute static_flag for ALL envs (||[vx, yaw]|| >= threshold => walking)
        self._static_flag = (
            torch.norm(self._commands, dim=1, keepdim=True) >= self.cfg.static_speed_threshold
        ).float()

    # ------------------------------------------------------------------ step lifecycle
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.clip(-1.0, 1.0)
        self._cur_freq = self._scale(actions[:, :2], self._phase_low, self._phase_high)         # 2 leg freqs
        joint_rate = self._scale(actions[:, 2:], self._joint_low, self._joint_high)             # 10 joint rates
        # drive the gait oscillator (once per policy step) and integrate incremental joint targets
        self._phase.set_frequency(self._cur_freq)
        self._phase.step(self.step_dt)
        self._joint_target = torch.clip(
            self._joint_target + joint_rate * self.step_dt, self._joint_limit_low, self._joint_limit_high
        )
        # command resampling timer
        self._command_time_left -= self.step_dt
        resample = (self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if resample.numel() > 0:
            self._resample_commands(resample)

    def _apply_action(self) -> None:
        # held constant across the 15-step decimation PD loop (matches RoboTamer's inner loop).
        # set_joint_position_target_index is the 3.0 API (set_joint_position_target is a deprecated alias).
        # NOTE: it is keyword-only (def ...(self, *, target, joint_ids=None, env_ids=None)).
        self.robot.set_joint_position_target_index(target=self._joint_target, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        pure = self._compute_pure_obs()
        # 3-frame stack, order [oldest, mid, newest]
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1] = pure
        policy_obs = self._obs_history.reshape(self.num_envs, -1)  # (N, 129)
        # privileged critic block (training-only superset; not part of the deploy contract)
        d = self.robot.data
        foot_z = wp.to_torch(d.body_pos_w)[:, self._foot_body_ids, 2]                         # (N, 2)
        if self._contact_sensor is not None:
            foot_force = torch.norm(
                wp.to_torch(self._contact_sensor.data.net_forces_w)[:, self._foot_contact_ids, :], dim=-1
            )                                                                                 # (N, 2)
        else:
            foot_force = torch.zeros_like(foot_z)                                             # contact-free: zeros (dims kept)
        priv = torch.cat(
            [
                wp.to_torch(d.root_lin_vel_b),                   # 3  privileged base lin vel
                wp.to_torch(d.root_pos_w)[:, 2:3],               # 1  base height
                foot_z,                                          # 2
                foot_force,                                      # 2
            ],
            dim=-1,
        )
        critic_obs = torch.cat([policy_obs, priv], dim=-1)  # (N, 137)
        return {"policy": policy_obs, "critic": critic_obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        lin_vel = wp.to_torch(d.root_lin_vel_b)
        ang_vel = wp.to_torch(d.root_ang_vel_b)
        roll, pitch, _ = euler_xyz_from_quat(wp.to_torch(d.root_quat_w))
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        base_h = wp.to_torch(d.root_pos_w)[:, 2]
        jv = wp.to_torch(d.joint_vel)[:, self._joint_ids]
        sf = self._static_flag.squeeze(-1)

        # tracking core (ungated)
        track_vx = torch.exp(-torch.square(self._commands[:, 0] - lin_vel[:, 0]) / 0.25)
        track_yaw = torch.exp(-torch.square(self._commands[:, 1] - ang_vel[:, 2]) / 0.25)
        r_height = torch.exp(-70.0 * torch.square(base_h - self.cfg.base_height_target))
        lateral = torch.square(lin_vel[:, 1])
        vertical = torch.square(lin_vel[:, 2])
        ang_xy = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
        orient = torch.sqrt(torch.square(roll) + torch.square(pitch) + 1e-8)

        # gait phasing: minimized when the two legs are exactly anti-phase
        sc = self._phase.sin_cos  # [sinL, sinR, cosL, cosR]
        foot_phase = torch.square(sc[:, 0] + sc[:, 1]) + torch.square(sc[:, 2] + sc[:, 3])

        # smoothness: 2nd difference (jerk) of joint targets and of frequency outputs
        act_smooth = torch.sum(
            torch.square(self._joint_target - 2.0 * self._prev_joint_target + self._prev2_joint_target), dim=1
        )
        freq_smooth = torch.sum(torch.square(self._cur_freq - 2.0 * self._prev_freq + self._prev2_freq), dim=1)

        # safety / regularization
        joint_vel_pen = torch.sum(torch.square(jv), dim=1)
        joint_torque_pen = torch.sum(torch.square(wp.to_torch(d.applied_torque)[:, self._joint_ids]), dim=1)

        # foot/contact: reward measured contact agreeing with the gait support/swing mask
        if self._contact_sensor is not None:
            foot_force = torch.norm(
                wp.to_torch(self._contact_sensor.data.net_forces_w)[:, self._foot_contact_ids, :], dim=-1
            )
            in_contact = foot_force > 1.0
            contact_match = torch.sum((in_contact == self._phase.support_mask).float(), dim=1) / 2.0
        else:
            contact_match = torch.zeros(self.num_envs, device=self.device)  # contact-free: term contributes 0
        # swing-foot clearance: reward lifting the swinging foot toward ~5 cm
        foot_z = wp.to_torch(d.body_pos_w)[:, self._foot_body_ids, 2]
        clearance = torch.sum(torch.clip(foot_z, 0.0, 0.05) * self._phase.swing_mask.float(), dim=1)

        rew = (
            self.cfg.rew_track_lin_vel * track_vx
            + self.cfg.rew_track_yaw * track_yaw
            + self.cfg.rew_base_height * r_height
            + self.cfg.rew_lateral_vel * lateral
            + self.cfg.rew_vertical_vel * vertical
            + self.cfg.rew_ang_vel_xy * ang_xy
            + self.cfg.rew_orientation * orient
            + self.cfg.rew_foot_phase * foot_phase * sf
            + self.cfg.rew_action_smooth * act_smooth
            + self.cfg.rew_freq_smooth * freq_smooth
            + self.cfg.rew_joint_vel * joint_vel_pen
            + self.cfg.rew_joint_torque * joint_torque_pen
            + self.cfg.rew_foot_contact * contact_match * sf
            + self.cfg.rew_foot_clearance * clearance * sf
            + self.cfg.rew_alive
        ) * self.step_dt
        # failure penalty (event, not rate-scaled)
        rew = rew + self.cfg.rew_termination * self._died.float()

        # roll the smoothness history (after using it)
        self._prev2_joint_target = self._prev_joint_target.clone()
        self._prev_joint_target = self._joint_target.clone()
        self._prev2_freq = self._prev_freq.clone()
        self._prev_freq = self._cur_freq.clone()
        return rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        roll, pitch, _ = euler_xyz_from_quat(wp.to_torch(self.robot.data.root_quat_w))
        roll, pitch = wrap_to_pi(roll), wrap_to_pi(pitch)
        tilted = (roll.abs() > self.cfg.max_tilt) | (pitch.abs() > self.cfg.max_tilt)
        fell = wp.to_torch(self.robot.data.root_pos_w)[:, 2] < self.cfg.base_height_termination
        # undesired contact on hip/knee links (base_link has no collision mesh -> use height above)
        if self._contact_sensor is not None:
            term_force = torch.norm(
                wp.to_torch(self._contact_sensor.data.net_forces_w)[:, self._term_contact_ids, :], dim=-1
            )
            contact_term = torch.any(term_force > 1.0, dim=1)
            died = tilted | fell | contact_term
        else:
            died = tilted | fell  # contact-free: rely on tilt + base height for the fall check
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._died = died
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device).long().flatten()

        # robot state -> default root + ref joint pose (wp.to_torch: 3.0 data is Warp)
        joint_pos = wp.to_torch(self.robot.data.default_joint_pos)[env_ids]
        joint_vel = wp.to_torch(self.robot.data.default_joint_vel)[env_ids]
        default_root = wp.to_torch(self.robot.data.default_root_state)[env_ids].clone()
        default_root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # task buffers (deploy order) -> start at the reference posture / rest frequency
        self._joint_target[env_ids] = self._ref_joint_pos[env_ids]
        self._prev_joint_target[env_ids] = self._ref_joint_pos[env_ids]
        self._prev2_joint_target[env_ids] = self._ref_joint_pos[env_ids]
        self._cur_freq[env_ids] = self.cfg.phase_rest_frequency
        self._prev_freq[env_ids] = self.cfg.phase_rest_frequency
        self._prev2_freq[env_ids] = self.cfg.phase_rest_frequency
        self._phase.reset(env_ids, randomize=True)          # random phase in training (det. eval = TODO)
        self._obs_history[env_ids] = 0.0                    # TODO(M2): seed with current pure obs to avoid 3-step transient
        self._resample_commands(env_ids)
