# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Tanh-mean Gaussian distribution — RoboTamer actor parity.

RoboTamer's actor squashes the MEAN during training: ``Normal(tanh(mu(x)), std)``. rsl_rl's stock
``GaussianDistribution`` uses the raw (unbounded) MLP output as the mean, which ratchets
exploration collapse (std shrinks, ``|mu|`` drifts >1); squashing the mean keeps std wide.

Squashes the mean ONLY (no SAC-style log-prob change-of-variables). The exported deterministic
module is tanh-bounded, matching the deploy contract (ONNX output = tanh(mu)).
"""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.modules.distribution import GaussianDistribution


class _TanhDeterministicOutput(nn.Module):
    """Export-friendly deterministic head: tanh(mlp_output) — the deploy-contract output."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)


class TanhMeanGaussianDistribution(GaussianDistribution):
    """Gaussian with tanh-squashed mean: ``Normal(tanh(mlp_output), std)`` (RoboTamer parity)."""

    def update(self, mlp_output: torch.Tensor) -> None:
        super().update(torch.tanh(mlp_output))

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return _TanhDeterministicOutput()
