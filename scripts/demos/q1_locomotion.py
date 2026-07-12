# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive demo for the Qmini (q1) biped — the walking counterpart of Isaac Lab's h1_locomotion.py.

Loads a trained policy (the bundled RoboTamer q2 reference by default) into the Qmini walking env and lets
you drive it. Click a robot to enter third-person follow; arrow keys steer the selected robot; Shift + drag
pushes any robot (native Isaac Sim viewport interaction). Best over WebRTC — launch with ``LIVESTREAM=2``.

.. code-block:: bash

    # inside the container, with WebRTC (connect the Isaac Sim WebRTC client to the server)
    LIVESTREAM=2 python scripts/demos/q1_locomotion.py
    # a different policy:
    LIVESTREAM=2 python scripts/demos/q1_locomotion.py --checkpoint logs/rsl_rl/qmini_birl/<run>/model_4999.pt

Controls (on the selected robot): UP forward · DOWN stop · LEFT / RIGHT turn · C toggle follow/free cam ·
ESC deselect. Shift + left-drag applies a force to any robot.
"""

import argparse
import os
import sys
from importlib import metadata

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  (scripts/rsl_rl/cli_args.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive Qmini (q1) locomotion demo.")
parser.add_argument("--task", type=str, default="Template-Qmini-Walk-1kHz-Play-v0", help="Registered task id.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of robots to spawn.")
cli_args.add_rsl_rl_args(parser)  # provides --checkpoint / --load_run / --resume / ...
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])  # demos open the Kit RTX viewport by default (WebRTC-friendly)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    args_cli.checkpoint = "models/golden_q2_rslrl.pt"  # default: the bundled golden reference policy

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import parse_env_cfg

import qmini_tasks.tasks  # noqa: F401  (registers the Qmini tasks)


class QminiLocomotionDemo:
    """Interactive Qmini walking demo: click-to-follow camera + arrow-key steering of the selected robot.

    A robot is selected by clicking it; only the selected robot is driven by the keyboard, the rest keep the
    env's own (frozen) command. Camera geometry is scaled for the small Qmini (base ~0.45 m vs H1 ~1 m)."""

    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

        env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)  # freeze: keyboard command holds
        env_cfg.episode_length_s = 1.0e6  # run continuously — no timeout resets during the demo
        self.env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        self.device = self.env.unwrapped.device

        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        # actor-only load: a golden/older checkpoint's critic shape may differ from the current env cfg.
        runner.load(retrieve_file_path(args_cli.checkpoint),
                    load_cfg={"actor": True, "critic": False, "optimizer": False, "rnd": False, "iteration": False})
        self.policy = runner.get_inference_policy(device=self.device)

        self._cmd_term = self.env.unwrapped.command_manager.get_term("base_velocity")
        self.create_camera()
        self.set_up_keyboard()
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-1.2, 0.0, 0.5], device=self.device)  # scaled for Qmini

    def create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi or not coi.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        T, R = 0.4, 0.5  # forward m/s, yaw rad/s — within the training command ranges
        self._key_to_cmd = {
            "UP": torch.tensor([T, 0.0, 0.0], device=self.device),
            "DOWN": torch.tensor([0.0, 0.0, 0.0], device=self.device),
            "LEFT": torch.tensor([T, 0.0, R], device=self.device),
            "RIGHT": torch.tensor([T, 0.0, -R], device=self.device),
            "ZEROS": torch.tensor([0.0, 0.0, 0.0], device=self.device),
        }

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._key_to_cmd and self._selected_id is not None:
                self._cmd_term.vel_command_b[self._selected_id] = self._key_to_cmd[event.input.name]
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C" and self._selected_id is not None:
                active = self.viewport.get_active_camera()
                self.viewport.set_active_camera(
                    self.perspective_path if active == self.camera_path else self.camera_path
                )
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if self._selected_id is not None:
                self._cmd_term.vel_command_b[self._selected_id] = self._key_to_cmd["ZEROS"]

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        paths = self._prim_selection.get_selected_prim_paths()
        if len(paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(paths) > 1:
            print("[demo] select exactly one robot")
        else:
            parts = paths[0].split("/")
            if len(parts) >= 4 and parts[3][0:4] == "env_":
                self._selected_id = int(parts[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("[demo] selected prim is not a robot")

    def _update_camera(self):
        pos = self.env.unwrapped.scene["robot"].data.root_pos_w.torch[self._selected_id, :]
        quat = self.env.unwrapped.scene["robot"].data.root_quat_w.torch[self._selected_id, :]
        eye = quat_apply(quat, self._camera_local_transform) + pos
        cam = ViewportCameraState(self.camera_path, self.viewport)
        cam.set_position_world(Gf.Vec3d(eye[0].item(), eye[1].item(), max(eye[2].item(), 0.25)), True)
        cam.set_target_world(Gf.Vec3d(pos[0].item(), pos[1].item(), pos[2].item() + 0.3), True)


def main():
    demo = QminiLocomotionDemo()
    obs, _ = demo.env.reset()
    while simulation_app.is_running():
        demo.update_selected_object()
        with torch.inference_mode():
            obs, _, _, _ = demo.env.step(demo.policy(obs))


if __name__ == "__main__":
    main()
    simulation_app.close()
