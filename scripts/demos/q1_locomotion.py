# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive Qmini (q1) locomotion demo — the walking counterpart of Isaac Lab's h1_locomotion.py.

Loads a trained policy (the bundled RoboTamer q2 reference by default) into the Qmini walking env,
launched through the same ``launch_simulation`` path as ``scripts/rsl_rl/play.py`` (the configuration
with a verified-working WebRTC picture on this setup). Run with ``LIVESTREAM=2`` and connect the
Isaac Sim WebRTC streaming client:

    LIVESTREAM=2 python scripts/demos/q1_locomotion.py
    LIVESTREAM=2 python scripts/demos/q1_locomotion.py --checkpoint logs/rsl_rl/qmini_birl/<run>/model_4999.pt

NOTE: the FIRST launch in a fresh container takes ~3-4 minutes before the first frame (RTX shader
compilation) — the stream is black until then. Subsequent launches are fast.

Controls: TAB select/cycle robot (or click one, if the client forwards mouse) · UP forward · DOWN stop ·
LEFT/RIGHT turn · P push the selected robot · C toggle cam · ESC deselect.
Env toggles: Q1_DEBUG=1 step/input echo · Q1_WINDOW=1 drop --no-window.
"""

import argparse
import os
import sys
from importlib import metadata

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  (scripts/rsl_rl/cli_args.py)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, parse_env_cfg

import qmini_tasks.tasks  # noqa: F401  (registers the Qmini tasks)

parser = argparse.ArgumentParser(description="Interactive Qmini (q1) locomotion demo.")
parser.add_argument("--task", type=str, default="Template-Qmini-Walk-200Hz-Play-v0", help="Registered task id.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of robots to spawn.")
cli_args.add_rsl_rl_args(parser)  # provides --checkpoint / --load_run / --resume / ...
add_launcher_args(parser)
args_cli = parser.parse_args()
if not args_cli.checkpoint:
    args_cli.checkpoint = "models/golden_q2_rslrl.pt"  # default: the bundled golden reference policy
# Silence two confirmed-benign, spammy log channels — real problems still surface as [Error]:
#   carb.omniclient.plugin  : OmniHub launch retries (we use no Nucleus; assets are local)
#   omni.physx.tensors.plugin: body-path probes from the flattened q1.usd's doubled-name / Physics-scope prims
args_cli.kit_args = (getattr(args_cli, "kit_args", "") or "") + \
    " --/log/channels/carb.omniclient.plugin=error --/log/channels/omni.physx.tensors.plugin=error" + \
    (" --/log/channels/omni.kit.livestream.webrtc.plugin=verbose" if os.environ.get("Q1_DEBUG") else "")
if os.environ.get("LIVESTREAM", "0") not in ("0", "") and not os.environ.get("Q1_WINDOW"):
    # livestream-only workaround: a phantom OS window drops frames and breaks client input routing.
    # Never set locally (it suppresses the desktop UI window); Q1_WINDOW=1 disables it under livestream too.
    args_cli.kit_args += " --no-window"

FWD, TURN = 0.6, 1.0  # forward m/s, yaw rad/s — near the max of the training command ranges (vx≤0.7, yaw≤1)


class QminiLocomotionDemo:
    """Click/TAB-to-follow camera + keyboard steering, ported from h1_locomotion.py for the small Qmini biped.

    A cyan sphere over each head is the click target; selecting one only moves the camera. The keyboard
    drives the selected robot; every robot walks forward by default. Steering commands live in
    ``self.commands`` and are asserted onto the env's velocity command each step. Kit-UI modules are
    imported here (the app is up only inside launch_simulation)."""

    def __init__(self, env):
        import carb
        import omni
        from omni.kit.viewport.utility import get_viewport_from_window_name
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf, Sdf, UsdGeom

        from isaaclab.sim.utils.stage import get_current_stage

        self._carb, self._Gf, self._ViewportCameraState = carb, Gf, ViewportCameraState
        self.env = env
        self.device = env.unwrapped.device
        self.robot = env.unwrapped.scene["robot"]
        self._cmd_term = env.unwrapped.command_manager.get_term("base_velocity")

        # every robot walks forward by default; only the selected one follows the keyboard
        self.commands = torch.zeros(env.unwrapped.num_envs, 3, device=self.device)
        self.commands[:, 0] = FWD

        stage = get_current_stage()
        # follow camera (the free camera stays the default /OmniverseKit_Persp)
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path, self.perspective_path = "/World/Camera", "/OmniverseKit_Persp"
        cam = stage.DefinePrim(self.camera_path, "Camera")
        cam.GetAttribute("focalLength").Set(8.5)
        coi = cam.GetProperty("omni:kit:centerOfInterest")
        if not coi or not coi.IsValid():
            cam.CreateAttribute("omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True,
                                Sdf.VariabilityUniform).Set(Gf.Vec3d(0, 0, -10))
        if self.viewport is not None:
            self.viewport.set_active_camera(self.perspective_path)

        # pick markers: big cyan spheres over each head — click targets, repositioned every step.
        # Under /World/envs/env_i so a click resolves to that env.
        self._markers = []
        for i in range(env.unwrapped.num_envs):
            s = UsdGeom.Sphere.Define(stage, f"/World/envs/env_{i}/pick_marker")
            s.CreateRadiusAttr(0.12)
            s.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.8, 1.0)])
            self._markers.append(UsdGeom.XformCommonAPI(s.GetPrim()))

        # keyboard + click selection
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        self._key_to_cmd = {
            "UP": (FWD, 0.0), "DOWN": (0.0, 0.0), "LEFT": (FWD, TURN), "RIGHT": (FWD, -TURN),
        }
        if os.environ.get("Q1_DEBUG"):  # echo mouse buttons + modifiers as the app receives them
            self._mouse = omni.appwindow.get_default_app_window().get_mouse()
            self._sub_mouse = self._input.subscribe_to_mouse_events(self._mouse, self._on_mouse_event)
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._kb_selected = None  # keyboard (TAB) selection — works even when the client forwards no mouse
        self._push_requested = False  # P key: shove the selected robot (root-velocity kick, official API)
        self._previous_selected_id = None
        self._camera_local = torch.tensor([-1.2, 0.0, 0.5], device=self.device)  # boom scaled for Qmini

    def update_pick_markers(self):
        local = self.robot.data.root_pos_w.torch - self.env.unwrapped.scene.env_origins  # (N,3) per-env frame
        for i, x in enumerate(self._markers):
            p = local[i]
            x.SetTranslate(self._Gf.Vec3d(p[0].item(), p[1].item(), p[2].item() + 0.35))  # float above the head

    def _on_mouse_event(self, event):
        carb = self._carb
        if event.type in (carb.input.MouseEventType.LEFT_BUTTON_DOWN, carb.input.MouseEventType.LEFT_BUTTON_UP,
                          carb.input.MouseEventType.RIGHT_BUTTON_DOWN):
            print(f"[q1] mouse {event.type} modifiers={getattr(event, 'modifiers', 'n/a')}", flush=True)

    def _on_keyboard_event(self, event):
        # Runs on the Kit input thread — only touch plain python/tensor state here.
        carb = self._carb
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        name = event.input.name
        if os.environ.get("Q1_DEBUG"):
            print(f"[q1] key: {name}", flush=True)
        if name == "TAB":  # cycle through robots — selection that needs no mouse
            n = self.env.unwrapped.num_envs
            self._kb_selected = 0 if self._kb_selected is None else (self._kb_selected + 1) % n
        elif name == "P":
            self._push_requested = True  # applied on the sim thread in apply_push()
        elif name == "ESCAPE":
            self._kb_selected = None
            self._prim_selection.clear_selected_prim_paths()
        elif self._selected_id is not None:
            if name in self._key_to_cmd:
                vx, yaw = self._key_to_cmd[name]
                self.commands[self._selected_id, 0] = vx
                self.commands[self._selected_id, 2] = yaw
            elif name == "C" and self.viewport is not None:
                active = self.viewport.get_active_camera()
                self.viewport.set_active_camera(
                    self.perspective_path if active == self.camera_path else self.camera_path
                )

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        # mouse click wins when present; otherwise fall back to the TAB (keyboard) selection
        paths = self._prim_selection.get_selected_prim_paths()
        env_part = None
        if len(paths) == 1:
            # find the ``env_<i>`` component anywhere in the clicked prim path (marker, robot body, ...)
            env_part = next((p for p in paths[0].split("/") if p.startswith("env_")), None)
        self._selected_id = int(env_part[4:]) if env_part is not None else self._kb_selected
        if self._selected_id is None:
            if self._previous_selected_id is not None:
                print("[demo] free camera", flush=True)
                if self.viewport is not None:
                    self.viewport.set_active_camera(self.perspective_path)
        else:
            if self._previous_selected_id != self._selected_id:
                print(f"[demo] following robot {self._selected_id}", flush=True)
                if self.viewport is not None:
                    self.viewport.set_active_camera(self.camera_path)
            self._update_camera()

    def apply_push(self):
        """Shove the selected robot: horizontal root-velocity kick — the same official API the training
        push randomization uses (mdp.push_by_setting_velocity), so it provably moves GPU-pipeline robots."""
        if not self._push_requested:
            return
        self._push_requested = False
        if self._selected_id is None:
            return
        rid = torch.tensor([self._selected_id], device=self.device)
        vel = self.robot.data.root_vel_w.torch[rid].clone()  # (1, 6) lin+ang
        theta = torch.rand(1, device=self.device) * 6.2831853
        vel[:, 0] += 0.7 * torch.cos(theta)
        vel[:, 1] += 0.7 * torch.sin(theta)
        self.robot.write_root_velocity_to_sim(vel, env_ids=rid)
        print(f"[demo] pushed robot {self._selected_id}", flush=True)

    def _update_camera(self):
        if self.viewport is None:
            return
        pos = self.robot.data.root_pos_w.torch[self._selected_id, :]
        quat = self.robot.data.root_quat_w.torch[self._selected_id, :]
        eye = quat_apply(quat, self._camera_local) + pos
        cam = self._ViewportCameraState(self.camera_path, self.viewport)
        cam.set_position_world(self._Gf.Vec3d(eye[0].item(), eye[1].item(), max(eye[2].item(), 0.25)), True)
        cam.set_target_world(self._Gf.Vec3d(pos[0].item(), pos[1].item(), pos[2].item() + 0.3), True)


def main():
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)  # freeze — we drive commands
    env_cfg.episode_length_s = 1.0e6  # run continuously

    with launch_simulation(env_cfg, args_cli):
        env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device)
        # actor-only load: a golden/older checkpoint's critic shape may differ from the current env cfg.
        runner.load(retrieve_file_path(args_cli.checkpoint),
                    load_cfg={"actor": True, "critic": False, "optimizer": False, "rnd": False, "iteration": False})
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        demo = QminiLocomotionDemo(env)
        obs, _ = env.reset()
        step = 0
        try:
            while True:
                demo.update_pick_markers()
                demo.update_selected_object()
                demo.apply_push()
                demo._cmd_term.vel_command_b[:] = demo.commands  # assert steering each step
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                if os.environ.get("Q1_DEBUG") and step % 200 == 0:
                    print(f"[q1] step {step}", flush=True)
                step += 1
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
