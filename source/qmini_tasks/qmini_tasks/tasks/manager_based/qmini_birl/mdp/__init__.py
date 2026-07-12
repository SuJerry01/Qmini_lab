# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Qmini BIRL manager-based task.

Gait-specific obs/action/reward terms are CUSTOM (faithful to RoboTamer); workflow-generic terms
(commands, terminations, reset event, alive/terminated) are reused from ``isaaclab.envs.mdp``.

Import the built-in terms EXPLICITLY (not ``import *``): the wildcard defeats the lazy_loader and
eagerly imports action terms that pull in ``pxr`` at module level. This mdp is imported during hydra
cfg resolution (before SimulationApp), where preloading pxr corrupts kit startup.
"""

from isaaclab.envs.mdp import (  # noqa: F401
    UniformVelocityCommandCfg,
    bad_orientation,
    illegal_contact,
    is_alive,
    is_terminated,
    reset_joints_by_offset,
    reset_root_state_uniform,
    reset_scene_to_default,
    root_height_below_minimum,
    time_out,
)

from .actions import *  # noqa: F401, F403
from .qmini_events import (  # noqa: F401  (no pxr)
    push_replace_vel_and_right,
    randomize_mass_inertia_tamer,
    randomize_pd_torque_gains,
)
# .commands is NOT imported here — velocity_command.py pulls pxr at module level, which corrupts kit
# during hydra cfg resolution (pre-SimulationApp). Reference it in cfgs by STRING instead:
#   class_type="qmini_tasks.tasks.manager_based.qmini_birl.mdp.commands:QminiUniformVelocityCommand"
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
