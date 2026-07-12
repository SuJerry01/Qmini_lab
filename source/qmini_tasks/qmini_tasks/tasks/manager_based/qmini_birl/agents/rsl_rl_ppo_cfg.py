# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl PPO config for the Qmini BIRL task — RoboTamer hyperparameters + asymmetric obs.

Two Tamer-parity customizations:
- ``TanhMeanGaussianDistribution``: actor squashes the mean (``Normal(tanh(mu), std)``); the stock
  linear mean collapses exploration.
- ``QminiPPO``: Tamer's adaptive-KL LR bounds (2e-5..1e-3) + joint actor+critic grad clip.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

_AGENTS = "qmini_tasks.tasks.manager_based.qmini_birl.agents"


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "qmini_birl"
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256],
        activation="relu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name=f"{_AGENTS}.tanh_gaussian:TanhMeanGaussianDistribution",
            init_std=0.8,
            std_type="scalar",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256],
        activation="relu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name=f"{_AGENTS}.qmini_ppo:QminiPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5.0e-4,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
    )


@configclass
class Qmini200HzPPORunnerCfg(PPORunnerCfg):
    """Same PPO, separate output dir so 200 Hz runs are labeled apart."""

    experiment_name = "qmini_birl_200hz"
