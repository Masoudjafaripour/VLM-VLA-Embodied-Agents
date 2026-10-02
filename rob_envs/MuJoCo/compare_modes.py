"""
Compare the three pick-and-place modes of jmujoco_jev.py (qwen / jev / combined)
over a fixed task set and several seeds, then plot the results.

    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/compare_modes.py                 # run all + plot
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/compare_modes.py --modes qwen --seeds 0 1
    isaac_venv/bin/python rob_envs/MuJoCo/compare_modes.py --plot-only                    # re-plot saved runs

Demos:   outputs/mujoco_jev/<mode>/seed<N>_<task>/{pick_place.mp4, vlm_input.png, final.png, result.json}
Summary: outputs/mujoco_jev/results.csv, outputs/mujoco_jev/comparison.png
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jmujoco_jev as J  # noqa: E402

# (instruction, expected pick, expected place). The last three need a little reasoning
# (paraphrase / negation), which keyword matching can't do.
TASKS = [
    ("put the red cube on the purple mat", "red cube", "purple mat"),
    ("place the yellow cube onto the white mat", "yellow cube", "white mat"),
    ("move the green cube to the purple mat", "green cube", "purple mat"),
    ("put the cube that is the color of a banana on the mat that is not white", "yellow cube", "purple mat"),
    ("the grass-colored block belongs on the white mat", "green cube", "white mat"),
    ("put the cube the color of a stop sign on the white mat", "red cube", "white mat"),
]

# Fixed categorical order (dataviz reference palette, slots 1-3: validated all-pairs)
MODE_ORDER = ["qwen", "jev", "combined"]
MODE_COLOR = {"qwen": "#2a78d6", "jev": "#eb6834", "combined": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def episode_plan(seeds, episodes):
    """(seed, task) pairs: every task for each seed in turn; with `episodes`, continue seeds until N are planned."""
    if episodes is None:
        return [(s, t) for s in seeds for t in TASKS]
    return [(i // len(TASKS), TASKS[i % len(TASKS)]) for i in range(episodes)]


def run(modes, seeds, episodes=None):
    vlm = J.VLM() if any(J.MODES[m][0] == "qwen" or J.MODES[m][1] == "qwen" for m in modes) else None
    for mode in modes:
        decider = J.Decider(J.MODES[mode][1], vlm)
        for seed, (task, pick, place) in episode_plan(seeds, episodes):
            print(f"\n===== {mode} | seed {seed} | {task}")
            try:
                J.run_episode(mode, task, seed, vlm=vlm, decider=decider, expected=(pick, place))
            except Exception as e:  # keep the sweep going; the failure shows up as a missing run
                print(f"[ERROR] {type(e).__name__}: {e}")


def load_results():
    rows = []
    for path in sorted(glob.glob(os.path.join(J.OUT_DIR, "*", "*", "result.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("expected_pick"):  # only benchmark runs (single CLI runs have no expected answer)
            rows.append(r)
    return rows


def write_csv(rows):
    path = os.path.join(J.OUT_DIR, "results.csv")
    keys = ["mode", "perception", "decider", "seed", "task", "expected_pick", "expected_place", "pick_true",
            "place_true", "decision_correct", "sim_success", "judged_success", "judge_agrees",
            "mean_cube_err_cm", "t_perceive_s", "t_decide_s", "t_verify_s", "escalated_decide", "escalated_judge",
            "out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            errs = [e for e in r["cube_err_cm"].values() if e is not None]
            w.writerow({**r, "mean_cube_err_cm": round(float(np.mean(errs)), 2) if errs else ""})
    return path


def rate(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return (100 * np.mean(vals), len(vals)) if vals else (None, 0)


def mode_label(mode, rows):
    deciders = {r["decider"] for r in rows}
    perception = "sim poses" if J.MODES[mode][0] == "gt" else "Qwen"
    names = {"cascade": "Jev System 1 / Qwen System 2", "keyword+qwen": "keyword System 1 / Qwen System 2"}
    decider = "/".join(sorted(names.get(d, d) for d in deciders))
    note = " (no API key: keyword fallback)" if any(d.startswith("keyword") for d in deciders) else ""
    esc, n = rate(rows, "escalated_decide")
    esc_txt = f", decision escalated in {esc:.0f}% of {n}" if esc is not None else ""
    return f"{mode}: {perception} → {decider}{note}{esc_txt}"


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_mode = {m: [r for r in rows if r["mode"] == m] for m in MODE_ORDER}
    modes = [m for m in MODE_ORDER if by_mode[m]]
    width = 0.8 / len(modes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    n_ep = {m: len(by_mode[m]) for m in modes}
    fig.suptitle("Pick-and-place: Qwen3-VL vs Jev vs combined", x=0.03, ha="left",
                 fontsize=14, color=INK, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLOR[m]) for m in modes]
    fig.legend(handles, [f"{mode_label(m, by_mode[m])} — {n_ep[m]} episodes" for m in modes],
               loc="upper left", bbox_to_anchor=(0.025, 0.955), ncol=1, frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=1.2, columnspacing=2)

    # A) Rates
    ax = axes[0, 0]
    metrics = [("decision_correct", "Decision\ncorrect"), ("sim_success", "Task success\n(sim)"),
               ("judge_agrees", "Self-check\nagrees w/ sim"), ("escalated_decide", "Escalated to\nSystem 2")]
    x = np.arange(len(metrics))
    for i, m in enumerate(modes):
        for j, (key, _) in enumerate(metrics):
            val, n = rate(by_mode[m], key)
            xpos = x[j] - 0.4 + width * (i + 0.5)
            if val is None:
                ax.text(xpos, 2, "n/a", ha="center", va="bottom", fontsize=8, color=MUTED, rotation=90)
                continue
            ax.bar(xpos, val, width, color=MODE_COLOR[m], edgecolor=SURFACE, linewidth=2,
                   label=m if j == 0 else None)
            ax.text(xpos, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 112)
    style(ax, "Success rates", "% of episodes")

    # B) Latency per stage (median, log scale)
    ax = axes[0, 1]
    stages = [("t_perceive_s", "Perception"), ("t_decide_s", "Decision"), ("t_verify_s", "Self-check")]
    x = np.arange(len(stages))
    for i, m in enumerate(modes):
        for j, (key, _) in enumerate(stages):
            vals = [r[key] for r in by_mode[m] if r[key] > 0]
            if not vals:
                continue
            med = float(np.median(vals))
            xpos = x[j] - 0.4 + width * (i + 0.5)
            ax.bar(xpos, med, width, color=MODE_COLOR[m], edgecolor=SURFACE, linewidth=2,
                   label=m if j == 0 else None)
            ax.text(xpos, med * 1.12, f"{med:.2f}s" if med < 10 else f"{med:.0f}s",
                    ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_yscale("log")
    ax.set_xticks(x, [label for _, label in stages])
    style(ax, "Latency per stage (median, log scale)", "seconds")

    # C) Task success per instruction x mode (sequential blue)
    ax = axes[1, 0]
    tasks = [t for t, _, _ in TASKS]
    grid = np.full((len(tasks), len(modes)), np.nan)
    for ti, t in enumerate(tasks):
        for mi, m in enumerate(modes):
            val, n = rate([r for r in by_mode[m] if r["task"] == t], "sim_success")
            if val is not None:
                grid[ti, mi] = val
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for ti in range(len(tasks)):
        for mi in range(len(modes)):
            v = grid[ti, mi]
            ax.text(mi, ti, "–" if np.isnan(v) else f"{v:.0f}%", ha="center", va="center", fontsize=9,
                    color=SURFACE if not np.isnan(v) and v > 60 else INK)
    ax.set_xticks(range(len(modes)), modes)
    ax.set_yticks(range(len(tasks)), [t if len(t) < 48 else t[:46] + "…" for t in tasks], fontsize=8)
    style(ax, "Task success by instruction")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)

    # D) Perception error (Qwen perception only; 'jev' uses simulator poses)
    ax = axes[1, 1]
    rng = np.random.default_rng(0)
    for i, m in enumerate(modes):
        errs = [e for r in by_mode[m] for e in r["cube_err_cm"].values() if e is not None]
        if J.MODES[m][0] == "gt":
            ax.text(i, 0.3, "sim poses\n(0 cm)", ha="center", va="bottom", fontsize=8.5, color=MUTED)
            continue
        if errs:
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(errs)), errs, s=22, color=MODE_COLOR[m],
                       edgecolor=SURFACE, linewidth=1, alpha=0.85)
            med = float(np.median(errs))
            ax.hlines(med, i - 0.25, i + 0.25, color=INK, linewidth=2)
            ax.text(i + 0.28, med, f"median {med:.1f} cm", va="center", fontsize=8, color=INK_2)
        missed = sum(e is None for r in by_mode[m] for e in r["cube_err_cm"].values())
        if missed:
            ax.text(i, -0.6, f"{missed} cube(s) missed", ha="center", fontsize=7.5, color=MUTED)
    ax.axhline(J.CUBE_HALF * 100, color=AXIS, linewidth=1, linestyle="--")
    ax.text(1, J.CUBE_HALF * 100 + 0.1, "half cube width", va="bottom", ha="center",
            fontsize=7.5, color=MUTED)
    ax.set_xticks(range(len(modes)), modes)
    ax.set_xlim(-0.6, len(modes) - 0.4)
    ax.set_ylim(bottom=-1)
    style(ax, "Cube localization error (xy)", "cm")

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path = os.path.join(J.OUT_DIR, "comparison.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def summary(rows):
    print(f"\n{'mode':10s} {'decider':9s} {'n':>3s} {'decision%':>10s} {'success%':>9s} {'judge-agree%':>13s}"
          f" {'t_decide(med)':>14s}")
    for m in MODE_ORDER:
        rs = [r for r in rows if r["mode"] == m]
        if not rs:
            continue
        fmt = lambda v: "n/a" if v[0] is None else f"{v[0]:.0f}"
        print(f"{m:10s} {'/'.join(sorted({r['decider'] for r in rs})):9s} {len(rs):3d} "
              f"{fmt(rate(rs, 'decision_correct')):>10s} {fmt(rate(rs, 'sim_success')):>9s} "
              f"{fmt(rate(rs, 'judge_agrees')):>13s} {np.median([r['t_decide_s'] for r in rs]):>13.3f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", choices=MODE_ORDER, default=MODE_ORDER)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, help="episodes per mode (cycles the tasks over seeds 0, 1, ...); "
                                                  "overrides --seeds")
    ap.add_argument("--plot-only", action="store_true", help="skip running, re-plot saved result.json files")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="run jev/combined with keyword matching when TYPESAFE_API_KEY is not set")
    args = ap.parse_args()

    needs_jev = [m for m in args.modes if J.MODES[m][1] in ("jev", "cascade")]
    if not args.plot_only and needs_jev and not os.environ.get("TYPESAFE_API_KEY") and not args.allow_fallback:
        sys.exit(f"TYPESAFE_API_KEY is not set but modes {needs_jev} need Jev. "
                 "Run `export TYPESAFE_API_KEY=...` first (or pass --allow-fallback).")

    if not args.plot_only:
        run(args.modes, args.seeds, args.episodes)
    rows = load_results()
    if not rows:
        sys.exit(f"No benchmark results under {J.OUT_DIR}")
    summary(rows)
    print(f"\nCSV  -> {write_csv(rows)}")
    print(f"Plot -> {plot(rows)}")


if __name__ == "__main__":
    main()
