# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the Qmini BIRL manager-based task (rsl_rl).

Four envs, all sharing the full-Tamer walking MDP (``QminiBirlEnvCfg``); the physics rate is explicit in
every id:

  Template-Qmini-Walk-1kHz-v0        1 kHz training (canonical, Tamer-faithful)
  Template-Qmini-Walk-1kHz-Play-v0   1 kHz play/view (randomization off)
  Template-Qmini-Walk-200Hz-v0       200 Hz training (A/B twin, ~5x throughput)
  Template-Qmini-Walk-200Hz-Play-v0  200 Hz play/view  — the default for play/demo

Both rates share the identical 66.7 Hz deploy interface (obs/action/control_dt 0.015); only the inner
physics substep rate differs. The 200 Hz variants log to a separate dir (experiment_name
"qmini_birl_200hz" via ``Qmini200HzPPORunnerCfg``). Env cfgs are lazy string entry points.
"""

import gymnasium as gym

from . import agents

# QminiBirlEnv = ManagerBasedRLEnv + RoboTamer total-reward floor.
gym.register(
    id="Template-Qmini-Walk-1kHz-v0",
    entry_point=f"{__name__}.qmini_birl_env:QminiBirlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_birl_env_cfg:QminiBirlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Qmini-Walk-1kHz-Play-v0",
    entry_point=f"{__name__}.qmini_birl_env:QminiBirlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_birl_env_cfg:QminiBirlPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Qmini-Walk-200Hz-v0",
    entry_point=f"{__name__}.qmini_birl_env:QminiBirlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_birl_env_cfg:QminiBirl200HzEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Qmini200HzPPORunnerCfg",
    },
)

gym.register(
    id="Template-Qmini-Walk-200Hz-Play-v0",
    entry_point=f"{__name__}.qmini_birl_env:QminiBirlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.qmini_birl_env_cfg:QminiBirl200HzPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Qmini200HzPPORunnerCfg",
    },
)
