# Copyright (c) 2026, The Qmini_lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Convert the q1 URDF to a cached USD (optional — QMINI_CFG also spawns the URDF on the fly).

Run inside the Isaac Lab container (needs Isaac Sim):

    ./isaaclab.sh -p scripts/convert_qmini_urdf.py
    # then in assets/qmini.py swap the spawn to:
    #   sim_utils.UsdFileCfg(usd_path="<printed path>", activate_contact_sensors=True, ...)

This is a thin wrapper over isaaclab.sim.converters.UrdfConverter so the conversion options match
QMINI_CFG (free base, merged fixed joints). The standalone scripts/tools/convert_urdf.py in Isaac
Lab does the same with CLI flags.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# --- launch the simulator (headless) before importing isaaclab.sim ---
parser = argparse.ArgumentParser(description="Convert the q1 (Qmini) URDF to USD.")
parser.add_argument("--output-dir", type=str, default=None, help="USD cache dir (default: alongside the URDF).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from qmini_tasks.assets.qmini import QMINI_URDF_PATH  # noqa: E402


def main() -> None:
    out_dir = args.output_dir or os.path.join(os.path.dirname(QMINI_URDF_PATH), "usd")
    os.makedirs(out_dir, exist_ok=True)
    cfg = UrdfConverterCfg(
        asset_path=QMINI_URDF_PATH,
        usd_dir=out_dir,
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        # PD drives are (re)set by the actuators in QMINI_CFG; use a neutral position drive here.
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=40.0, damping=1.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print("\n" + "=" * 70)
    print(f"[Qmini] URDF : {QMINI_URDF_PATH}")
    print(f"[Qmini] USD  : {converter.usd_path}")
    print("Set assets/qmini.py spawn -> sim_utils.UsdFileCfg(usd_path=<above>, activate_contact_sensors=True)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
