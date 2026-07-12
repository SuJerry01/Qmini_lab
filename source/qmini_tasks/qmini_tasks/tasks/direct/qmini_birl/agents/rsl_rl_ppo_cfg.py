# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl PPO config for the Qmini BIRL Direct task.

Reproduces RoboTamer4Qmini's hand-written PPO hyperparameters (config/Base.py ``algorithm`` + the
learnable scalar action std) on rsl-rl 5.0.1, with an **asymmetric critic** and **no observation
normalization** (the deploy ONNX contract has no normalizer). See ``docs/deep_dive.md`` §3 for the
full RoboTamer->rsl_rl mapping and the behavioural caveats.

Uses the auto-converted ``policy = RslRlPpoActorCriticCfg(...)`` style + ``obs_groups`` (mirrors the
shipped/working asymmetric examples). Class name is ``PPORunnerCfg`` to match the gym registration.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # runner (RoboTamer config/Base.py runner: num_steps_per_env 24, max_iterations 5000, save 200)
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 200
    experiment_name = "qmini_birl_direct"
    clip_actions = 1.0  # actions clipped to [-1,1] before scale_transform (matches the env)

    # asymmetric actor/critic: actor sees env["policy"] (129), critic sees env["critic"] (137 superset)
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    # actor/critic networks — separate MLPs (512,256) ReLU, scalar state-independent std init 0.8,
    # NO obs normalization (deploy contract). (RoboTamer rl/module/continuous.py + config/Base.py policy)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        noise_std_type="scalar",
        state_dependent_std=False,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256],
        critic_hidden_dims=[512, 256],
        activation="relu",
    )

    # PPO algorithm — RoboTamer config/Base.py algorithm values.
    # Note: clip_param=0.2 governs BOTH the policy surrogate and the value clip (as in RoboTamer);
    # advantage normalized globally (not per-minibatch). See deep_dive.md migration watch-list.
    algorithm = RslRlPpoAlgorithmCfg(
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
