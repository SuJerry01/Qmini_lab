# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Custom termination terms for the Qmini BIRL gait (RoboTamer's jact_over).

Terminate (training-only) when any joint is pinned at its limit — both the commanded target AND the
measured position within 0.02 rad of the same limit. Kills the "stuck straight-knee crouch" optimum.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from qmini_tasks.assets.qmini import QMINI_POS_LIMIT_HIGH, QMINI_POS_LIMIT_LOW

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_at_limit(
    env: ManagerBasedRLEnv,
    action_name: str = "birl_action",
    tol: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """RoboTamer ``jact_over``: True if ANY joint has both its commanded target and its measured
    position within ``tol`` (0.02) of the SAME position limit (deploy-contract joint pos limits).
    """
    term = env.action_manager.get_term(action_name)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos.torch[:, term.joint_ids]          # (N,10) deploy order
    act = term.current_joint_act                                      # (N,10) target
    low = torch.tensor(QMINI_POS_LIMIT_LOW, device=env.device)        # (10,)
    high = torch.tensor(QMINI_POS_LIMIT_HIGH, device=env.device)

    at_low = ((act - low).abs() < tol) & ((q - low).abs() < tol)
    at_high = ((act - high).abs() < tol) & ((q - high).abs() < tol)
    return (at_low | at_high).any(dim=1)                              # (N,) bool
