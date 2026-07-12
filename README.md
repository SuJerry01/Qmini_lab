# Qmini_lab — Isaac Lab port of RoboTamer4Qmini

Isaac Lab port of the Unitree **Qmini** biped locomotion controller (migrated from Isaac Gym). One
manager-based task, **`Template-Qmini-Walk-1kHz-v0`**, trained with **RSL-RL**. Everything runs inside the
project's Docker container.

## Get the code

```bash
git clone https://github.com/SuJerry01/Qmini_lab.git
cd Qmini_lab && git lfs pull      # fetch the LFS assets: q1.usd + the bundled golden policy
```

## Setup

Build the container image (once):

```bash
cd docker && docker compose --env-file .env.base build && cd ..
```

Open a shell in the container — **run this from the repo root** (`$(pwd)` is the mount source), then run
every command below inside it:

```bash
docker run --rm -it --gpus all --network host \
  -e OMNI_KIT_ALLOW_ROOT=1 -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -v "$(pwd)":/workspace/Qmini_lab -w /workspace/Qmini_lab \
  --entrypoint bash qmini-lab:3.0.0-beta2-post1
```

Add `-e LIVESTREAM=2` for the WebRTC live view.

Confirm the task is registered:

```bash
python scripts/list_envs.py --keyword Qmini
```

## Train

```bash
python scripts/rsl_rl/train.py --task Template-Qmini-Walk-1kHz-v0 --headless
```

Checkpoints and logs are written to `logs/rsl_rl/qmini_birl/<timestamp>/`.

## Play

Play/view defaults to the **200 Hz** variant (faster; identical 66.7 Hz control interface as 1 kHz). To view
at the exact training rate, swap `200Hz` → `1kHz` in the task id.

```bash
# play a trained checkpoint with 32 environments
python scripts/rsl_rl/play.py --task Template-Qmini-Walk-200Hz-Play-v0 --num_envs 32 \
  --checkpoint logs/rsl_rl/qmini_birl/<run>/model_4999.pt

# record a video of a trained agent (requires ffmpeg)
python scripts/rsl_rl/play.py --task Template-Qmini-Walk-200Hz-Play-v0 --headless --video --video_length 200 \
  --checkpoint logs/rsl_rl/qmini_birl/<run>/model_4999.pt

# Baseline (no training needed): models/golden_q2_rslrl.pt is RoboTamer's ORIGINAL trained policy — its
# q2 checkpoint converted to the rsl_rl format — bundled as the migration parity baseline. Play it to see
# the target gait, and compare any newly-trained policy against it (golden-parity).
python scripts/rsl_rl/play.py --task Template-Qmini-Walk-200Hz-Play-v0 --num_envs 1 \
  --checkpoint models/golden_q2_rslrl.pt
```

## Remote visualization

Watch a run on a headless server without a local display:

```bash
# lightweight scene view in your browser — Rerun or Viser (the URL is printed on startup)
python scripts/rsl_rl/play.py --task Template-Qmini-Walk-200Hz-Play-v0 --num_envs 1 \
  --checkpoint models/golden_q2_rslrl.pt --viz rerun          # or: --viz viser

# training curves
tensorboard --logdir logs/rsl_rl
```

**Interactive demo** (`scripts/demos/q1_locomotion.py` — the Qmini counterpart of Isaac Lab's
`h1_locomotion.py`): load a policy and drive/push robots live to eyeball behaviour by hand. Needs the full
RTX viewport, so run over WebRTC (connect the Isaac Sim WebRTC streaming client to the server):

```bash
LIVESTREAM=2 python scripts/demos/q1_locomotion.py                                  # bundled golden policy
LIVESTREAM=2 python scripts/demos/q1_locomotion.py --checkpoint logs/rsl_rl/qmini_birl/<run>/model_4999.pt
```

Controls: **TAB selects/cycles robots** (or click one) · arrow keys steer the selected robot · **P pushes
the selected robot** · C toggles the camera · ESC deselects · Shift + left-drag shove (native Isaac Sim
physics mouse interaction) additionally requires a client that forwards mouse input. `play.py` itself stays
non-interactive — the demo is the interactive one (official Isaac Lab keeps play and demos separate).
