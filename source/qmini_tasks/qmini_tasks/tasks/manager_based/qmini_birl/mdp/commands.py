# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Qmini velocity command with RoboTamer's sub-threshold zeroing.

Stock ``UniformVelocityCommand`` leaves ~3.5% of env-time at ``0 < ||cmd|| < 0.15``, where gait rewards
are gated OFF but velocity tracking still demands motion (trains a shuffle). RoboTamer zeroes (a) the
whole command when ``||[vx,yaw]|| < 0.15`` and (b) each component when ``|component| < 0.15``, so every
env is either walking or exactly standing.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


class QminiUniformVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand + Tamer sub-threshold zeroing (whole-command and per-component)."""

    THRESH = 0.15

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        cmd = self.vel_command_b[env_ids]                      # (n, 3) = [vx, vy, yaw]
        # (a) zero the whole command if the planar [vx, yaw] magnitude is sub-threshold
        whole = torch.norm(cmd[:, [0, 2]], dim=1, keepdim=True) >= self.THRESH
        cmd *= whole.float()
        # (b) zero each commanded component independently if sub-threshold
        cmd[:, 0] *= (cmd[:, 0].abs() >= self.THRESH).float()
        cmd[:, 2] *= (cmd[:, 2].abs() >= self.THRESH).float()
        self.vel_command_b[env_ids] = cmd
