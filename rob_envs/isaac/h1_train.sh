#!/usr/bin/env bash
# Train Unitree H1 walking (RSL-RL PPO, Isaac-Velocity-Flat-H1) with live learning-curve plots.
#   bash rob_envs/isaac/h1_train.sh                 # 4096 envs, default iterations
#   bash rob_envs/isaac/h1_train.sh 1024 300        # num_envs, max_iterations
# Logs/checkpoints: rob_envs/isaac/logs/rsl_rl/h1_flat/<timestamp>/   (curves: training_curves.png)
cd "$(dirname "$0")"
ROOT=../..
NUM_ENVS=${1:-4096}
ITERS=${2:+--max_iterations $2}

START=$(date +%s)
"$ROOT/isaac_venv/bin/isaaclab" train --rl_library rsl_rl --task Isaac-Velocity-Flat-H1 \
  --num_envs "$NUM_ENVS" $ITERS &
TRAIN_PID=$!

# wait for this run's log dir, then plot it every 15 s until training ends
until RUN=$(find logs/rsl_rl/h1_flat -mindepth 1 -maxdepth 1 -type d -newermt "@$START" 2>/dev/null | head -1) && [ -n "$RUN" ]; do
  kill -0 $TRAIN_PID 2>/dev/null || exit 1
  sleep 5
done
"$ROOT/isaac_venv/bin/python" plot_training.py --run "$RUN" &
PLOT_PID=$!

wait $TRAIN_PID
kill $PLOT_PID 2>/dev/null
"$ROOT/isaac_venv/bin/python" plot_training.py --run "$RUN" --once   # final plot
