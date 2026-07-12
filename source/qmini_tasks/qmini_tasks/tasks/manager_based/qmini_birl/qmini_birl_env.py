# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""``ManagerBasedRLEnv`` subclass adding RoboTamer's reward shaping.

- per-term clip: ``clip(weighted_term, -4, 5) * dt`` before summing (RoboTamer ``birl_task.py:429-430``);
  stops any one unbounded penalty from dominating the step.
- sum floor: ``rew = clip(Σ terms, min=0)`` (RoboTamer ``gym_env_wrapper.py:66``); RoboTamer has no
  terminal penalty, so staying alive is never worse than dying.

Resolved lazily at ``gym.make`` (string entry point) after the Isaac Sim app launches.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardManager


class QminiRewardManager(RewardManager):
    """RewardManager clipping each WEIGHTED term to ``[-4, 5]`` before ``*dt`` and summing (RoboTamer birl_task.py:429-430)."""

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # weighted term -> per-term clip [-4,5] -> *dt  (RoboTamer birl_task.py:429-430)
            weighted = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
            value = torch.clip(weighted, -4.0, 5.0) * dt
            self._reward_buf += value
            self._episode_sums[name] += value
            self._step_reward[:, term_idx] = value / dt
        return self._reward_buf


class QminiBirlEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + RoboTamer per-term reward clip + total-reward floor."""

    def load_managers(self):
        super().load_managers()
        # swap in the per-term-clipping reward manager (RoboTamer birl_task.py:429-430);
        # safe post-super: nothing after the reward-manager line in load_managers reads it.
        self.reward_manager = QminiRewardManager(self.cfg.rewards, self)

    def step(self, action: torch.Tensor):
        obs, rew, terminated, truncated, extras = super().step(action)
        rew.clamp_(min=0.0)  # RoboTamer sum floor (gym_env_wrapper.py:66): no terminal penalty
        return obs, rew, terminated, truncated, extras
