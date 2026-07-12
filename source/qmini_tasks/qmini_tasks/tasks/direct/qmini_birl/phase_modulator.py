# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Vectorized per-leg gait phase oscillator (Isaac Lab reconstruction of RoboTamer's PhaseModulator).

**What** — holds a per-env, per-leg phase ``phi in [0, 2*pi)`` and the angular frequency the policy
commands for it. Each policy step it integrates ``phi += 2*pi * f * dt`` and exposes:
``sin/cos(phi)`` (the gait clock the actor observes) and a recentred ``f*0.3 - 1`` frequency feature.

**Why** — this is the CPG-like inductive bias at the heart of the BIRL gait: the policy only tunes a
rhythm (2 frequencies) and shapes the legs around a *structurally guaranteed* periodic, drift-free
phase, instead of synthesising a clock open-loop (fragile, drifts, breaks under latency). The
support/swing split at ``convert_phi`` (~60% stance duty) lets the env couple this clock to the foot
rewards. See ``docs/deep_dive.md`` §1.4.

**How it differs from the source (not a verbatim port)** — it owns *no* simulation handles, only
torch buffers, so it is a clean component a ``DirectRLEnv`` updates once per policy step in
``_pre_physics_step`` (it could equally back a custom ``isaaclab.managers.CommandTerm``). State is
reset per-env through ``reset(env_ids)`` to integrate with Isaac Lab's partial-reset model.
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

TWO_PI = 2.0 * math.pi


class PhaseModulator:
    """Per-leg phase oscillator driven by policy frequency commands.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device.
        num_legs: Number of legs (2 for a biped).
        rest_frequency: Frequency the oscillator resets to [Hz-like]. RoboTamer rests at 0.5.
        convert_phi: Phase boundary [rad] splitting stance (``phi < convert_phi``) from swing.
            RoboTamer uses ``1.2*pi`` → ~60% stance duty factor.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        num_legs: int = 2,
        rest_frequency: float = 0.5,
        convert_phi: float = 1.2 * math.pi,
    ):
        self.num_envs = num_envs
        self.device = device
        self.num_legs = num_legs
        self.rest_frequency = rest_frequency
        self.convert_phi = convert_phi
        # state buffers
        self.phase = torch.zeros(num_envs, num_legs, device=device)
        self.frequency = torch.full((num_envs, num_legs), rest_frequency, device=device)

    def set_frequency(self, frequency: torch.Tensor) -> None:
        """Set the commanded per-leg frequency (the policy's first 2 action channels, de-scaled)."""
        self.frequency = frequency

    def step(self, dt: float) -> torch.Tensor:
        """Advance the phase by one policy step: ``phi = (phi + 2*pi*f*dt) mod 2*pi``."""
        self.phase = torch.remainder(self.phase + TWO_PI * self.frequency * dt, TWO_PI)
        return self.phase

    def reset(self, env_ids: Sequence[int] | torch.Tensor, randomize: bool = True) -> None:
        """Reset phase/frequency for the given envs.

        Args:
            env_ids: Environments to reset.
            randomize: If True (training), randomize phase in ``[0, 2*pi)``; if False
                (eval/deploy), start both legs at phase 0 for determinism — matches
                ``PhaseModulator.reset(render=...)`` in the source.
        """
        n = len(env_ids) if not isinstance(env_ids, torch.Tensor) else env_ids.numel()
        if randomize:
            self.phase[env_ids] = torch.rand(n, self.num_legs, device=self.device) * TWO_PI
        else:
            self.phase[env_ids] = 0.0
        self.frequency[env_ids] = self.rest_frequency

    @property
    def sin_cos(self) -> torch.Tensor:
        """``[sin(phi_L), sin(phi_R), cos(phi_L), cos(phi_R)]`` — shape ``(num_envs, 4)``.

        Smooth, wrap-free encoding of where each leg is in its cycle (the 4 ``pm_phase`` obs dims).
        Order matches the deploy C++ side (``rl_controller.cpp``).
        """
        return torch.cat([torch.sin(self.phase), torch.cos(self.phase)], dim=-1)

    @property
    def freq_feature(self) -> torch.Tensor:
        """``f*0.3 - 1`` — shape ``(num_envs, 2)``. The recentred frequency fed back to the policy."""
        return self.frequency * 0.3 - 1.0

    @property
    def support_mask(self) -> torch.Tensor:
        """Per-leg stance mask ``(num_envs, 2)`` bool: ``0 <= phi < convert_phi`` (~60% of the cycle)."""
        return (self.phase >= 0.0) & (self.phase < self.convert_phi)

    @property
    def swing_mask(self) -> torch.Tensor:
        """Per-leg swing mask ``(num_envs, 2)`` bool (complement of :attr:`support_mask`)."""
        return ~self.support_mask
