# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""RoboTamer's sensor model: fixed per-env bias + 1 kHz ring buffers + GLOBAL random read delay.

Faithful port of the chain the golden ``q2`` policy was trained with (``randomize_noise=true,
use_state_filter=false``):

1. Bias, not white noise: each env draws ONE uniform offset per channel at startup (±dof_pos 0.1,
   ±dof_vel 1.2, ±euler 0.15, ±ang_vel 0.3, ±base_acc 3.0) and keeps it. No low-pass (state filter off)
   → buffers store ``raw + bias``.
2. Buffers tick at the SIM substep (1 kHz) → delay steps are milliseconds.
3. The read delay is a GLOBAL scalar shared by ALL envs, one per group — joints U[10,40] ms, angle/rate
   U[20,50] ms — re-drawn every 200 control steps (~3 s).
4. ``base_acc`` = IMU model: finite-diff of base lin vel at sim rate, base frame, clip ±30, ``z += 9.8``
   (proper accel), then biased and clipped ±30 again on append.
5. On reset the buffer is flooded with the CURRENT raw (un-biased) values.

``delay(step)`` timing: ``delay(1)`` = newest frame, ``delay(d)`` = ``d-1`` substeps old. Appends happen
in ``QminiBirlAction.apply_actions`` (once per substep, before ``sim.step``): the newest frame is the
END of the previous substep, a fixed 1 ms offset that is noise-level under the 10-50 ms delay.
"""

from __future__ import annotations

import random
import torch

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi

# Tamer noise_values — uniform bias half-ranges.
TAMER_NOISE = {
    "joint_pos": 0.1,   # dof_pos
    "joint_vel": 1.2,   # dof_vel
    "base_euler": 0.15,  # "gravity"
    "base_ang_vel": 0.3,  # ang_vel
    "base_acc": 3.0,
}


class SensorDelayBuffers:
    """Per-substep sensor ring buffers with fixed per-env bias and a global randomized read delay."""

    CHANNELS = {"joint_pos": 10, "joint_vel": 10, "base_euler": 3, "base_ang_vel": 3, "base_acc": 3}

    def __init__(
        self,
        num_envs: int,
        device: str,
        sim_dt: float,
        delay_enabled: bool = True,
        noise_enabled: bool = True,
        delay_joint_range: tuple[int, int] = (10, 40),
        delay_rate_range: tuple[int, int] = (20, 50),
        delay_angle_range: tuple[int, int] = (20, 50),
        resample_every_steps: int = 200,
        maxlen: int = 64,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self.sim_dt = sim_dt
        self.delay_enabled = delay_enabled
        self.maxlen = maxlen
        self._ranges = {
            "joint": delay_joint_range,
            "rate": delay_rate_range,
            "angle": delay_angle_range,
        }
        self._resample_every = resample_every_steps
        # ring buffers (N, K, D), one write pointer for all channels
        self._buf = {
            name: torch.zeros(num_envs, maxlen, dim, device=device) for name, dim in self.CHANNELS.items()
        }
        self._ptr = 0  # slot last written
        self._primed = False  # first append floods the ring (pre-first-reset safety)
        # fixed per-env bias, drawn ONCE (never re-drawn)
        self.bias = {
            name: (torch.rand(num_envs, dim, device=device) * 2.0 - 1.0) * (TAMER_NOISE[name] if noise_enabled else 0.0)
            for name, dim in self.CHANNELS.items()
        }
        # IMU finite-difference state + the freshest RAW (un-biased) proper acceleration
        self._last_lvel_w = torch.zeros(num_envs, 3, device=device)
        self.last_raw_acc = torch.zeros(num_envs, 3, device=device)
        # global scalar delays (substeps), Tamer semantics: shared by every env
        self._delays = {"joint": 1, "rate": 1, "angle": 1}
        if delay_enabled:
            self._draw_delays()

    # ------------------------------------------------------------------ delay bookkeeping
    def _draw_delays(self) -> None:
        for key, (lo, hi) in self._ranges.items():
            self._delays[key] = random.randint(lo, hi)

    def maybe_resample(self, common_step_counter: int) -> None:
        """Re-draw the global delays every ``resample_every_steps`` CONTROL steps."""
        if self.delay_enabled and common_step_counter % self._resample_every == 0:
            self._draw_delays()

    # ------------------------------------------------------------------ write path (per substep)
    def append(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
    ) -> None:
        """Push one substep of raw sensor data (biased on write). Call once per physics substep."""
        roll, pitch, yaw = euler_xyz_from_quat(root_quat_w)
        euler = torch.stack([wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)], dim=-1)
        # IMU proper acceleration
        acc_b = quat_apply_inverse(root_quat_w, (root_lin_vel_w - self._last_lvel_w) / self.sim_dt)
        acc_b = acc_b.clip(-30.0, 30.0)
        acc_b[:, 2] += 9.8
        self._last_lvel_w[:] = root_lin_vel_w
        self.last_raw_acc[:] = acc_b
        values = {
            "joint_pos": joint_pos + self.bias["joint_pos"],
            "joint_vel": joint_vel + self.bias["joint_vel"],
            "base_euler": euler + self.bias["base_euler"],
            "base_ang_vel": root_ang_vel_b + self.bias["base_ang_vel"],
            "base_acc": (acc_b + self.bias["base_acc"]).clip(-30.0, 30.0),  # clip again after bias
        }
        if not self._primed:  # flood before the first real frame so early reads are sane
            for name, v in values.items():
                self._buf[name][:] = v.unsqueeze(1)
            self._primed = True
            self._ptr = 0
            return
        self._ptr = (self._ptr + 1) % self.maxlen
        for name, v in values.items():
            self._buf[name][:, self._ptr] = v

    # ------------------------------------------------------------------ read path (per control step)
    def _read(self, name: str, group: str) -> torch.Tensor:
        # DelayDeque.delay(d): d=1 -> newest, d -> d-1 substeps old
        d = self._delays[group] if self.delay_enabled else 1
        return self._buf[name][:, (self._ptr - (d - 1)) % self.maxlen]

    def delayed_joint_pos(self) -> torch.Tensor:
        return self._read("joint_pos", "joint")

    def delayed_joint_vel(self) -> torch.Tensor:
        return self._read("joint_vel", "joint")

    def delayed_base_euler(self) -> torch.Tensor:
        return self._read("base_euler", "angle")

    def delayed_base_ang_vel(self) -> torch.Tensor:
        return self._read("base_ang_vel", "rate")

    def delayed_base_acc(self) -> torch.Tensor:
        return self._read("base_acc", "rate")

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        env_ids: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
    ) -> None:
        """Flood the ring with the CURRENT raw values for ``env_ids`` (Tamer fills un-biased on reset)."""
        roll, pitch, yaw = euler_xyz_from_quat(root_quat_w)
        euler = torch.stack([wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)], dim=-1)
        self._last_lvel_w[env_ids] = root_lin_vel_w
        self.last_raw_acc[env_ids] = 0.0
        self.last_raw_acc[env_ids, 2] = 9.8
        fills = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "base_euler": euler,
            "base_ang_vel": root_ang_vel_b,
            "base_acc": self.last_raw_acc[env_ids],
        }
        for name, v in fills.items():
            self._buf[name][env_ids] = v.unsqueeze(1)
