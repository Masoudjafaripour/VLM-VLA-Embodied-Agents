#!/usr/bin/env bash
# Unitree H1 humanoid walking with NVIDIA's pretrained RSL-RL policy (Isaac-Velocity-Flat-H1).
#   bash rob_envs/isaac/h1_walk.sh            # GUI
#   bash rob_envs/isaac/h1_walk.sh --video    # headless, records an mp4
# Train your own instead:  isaaclab train --rl_library rsl_rl --task Isaac-Velocity-Flat-H1 --num_envs 4096
cd "$(dirname "$0")"
ROOT=../..
if [ "$1" == "--video" ]; then
  EXTRA="--video --video_length 300"
else
  EXTRA="--viz kit"
fi
"$ROOT/isaac_venv/bin/isaaclab" play --rl_library rsl_rl --task Isaac-Velocity-Flat-H1 \
  --num_envs 4 --checkpoint pretrained $EXTRA
