"""Live learning curves for Isaac Lab RSL-RL training (matplotlib Agg backend, no display needed).

Reads the TensorBoard event file of a run and re-saves a PNG every --interval seconds.
Open the PNG in VS Code / an image viewer; it refreshes as training progresses.

Run (in a 2nd terminal while training):
    python rob_envs/isaac/plot_training.py                      # latest run under rob_envs/isaac/logs/rsl_rl
    python rob_envs/isaac/plot_training.py --run <run_dir> --once
"""
import argparse
import glob
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [  # (title, [tags])
    ("Mean reward", ["Train/mean_reward"]),
    ("Mean episode length", ["Train/mean_episode_length"]),
    ("Velocity tracking error", ["Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw"]),
    ("Tracking rewards", ["Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/track_ang_vel_z_exp"]),
    ("Losses", ["Loss/value", "Loss/surrogate"]),
    ("Terminations / policy std", ["Episode_Termination/base_contact", "Episode_Termination/time_out", "Policy/mean_std"]),
]


def latest_run(root):
    runs = [d for d in glob.glob(os.path.join(root, "*", "*")) if glob.glob(os.path.join(d, "events.out*"))]
    return max(runs, key=os.path.getmtime) if runs else None


def plot(run, out):
    ea = EventAccumulator(run, size_guidance={"scalars": 0})  # 0 = keep all points
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    n_iter = 0
    for ax, (title, panel_tags) in zip(axes.flat, PANELS):
        for tag in panel_tags:
            if tag in tags:
                ev = ea.Scalars(tag)
                ax.plot([e.step for e in ev], [e.value for e in ev], label=tag.split("/")[-1])
                n_iter = max(n_iter, len(ev))
        ax.set_title(title)
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle(f"{os.path.relpath(run, HERE)}   ({n_iter} iterations)")
    fig.tight_layout()
    fig.savefig(out + ".tmp.png", dpi=100)
    os.replace(out + ".tmp.png", out)  # atomic swap so viewers never see a half-written file
    plt.close(fig)
    return n_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run dir with events.out.tfevents.* (default: latest)")
    parser.add_argument("--logs", default=os.path.join(HERE, "logs", "rsl_rl"))
    parser.add_argument("--out", help="output PNG (default: <run>/training_curves.png)")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between refreshes")
    parser.add_argument("--once", action="store_true", help="plot once and exit")
    args = parser.parse_args()

    run = args.run
    while run is None:  # training may not have created its log dir yet
        run = latest_run(args.logs)
        if run is None:
            print(f"waiting for a run in {args.logs} ...")
            time.sleep(args.interval)
    out = args.out or os.path.join(run, "training_curves.png")
    print(f"plotting {run}\n  -> {out}", flush=True)

    while True:
        n = plot(run, out)
        print(f"[{time.strftime('%H:%M:%S')}] updated ({n} iterations)", flush=True)
        if args.once:
            break
        time.sleep(args.interval)
