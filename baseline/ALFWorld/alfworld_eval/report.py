"""Build a paper-style ALFWorld results table (success %, by task type) as Markdown.

Reads the per-episode *.jsonl trajectory files in --out-dir (so it also works on
runs still in progress) and writes RESULTS.md. If one model/split has several
runs, the newest file is used.

  Avg. = unweighted mean of the 6 task-type success rates (how paper tables report it)
  All  = success rate over all episodes (weighted by task-type counts)
"""

import argparse
import glob
import json
import os
import re

import numpy as np

TYPES = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2"]

# Reported "Base Model" rows from the reference paper table, for side-by-side comparison.
REFERENCE = {
    "Qwen/Qwen3-1.7B": [6.8, 60.2, 0.0, 0.0, 3.6, 4.1, 12.4],
}


def load_runs(out_dir, split):
    runs = {}
    for path in sorted(glob.glob(os.path.join(out_dir, f"*_{split}_s*.jsonl"))):  # sorted => newest last
        eps = [json.loads(l) for l in open(path) if l.strip()]
        if eps and "task_type" in eps[0]:
            model = re.sub(rf"_{split}_s\d+_\d{{8}}-\d{{6}}\.jsonl$", "", os.path.basename(path)).replace("_", "/", 1)
            runs[model] = (path, eps)
    return runs


def size_key(model):
    m = re.search(r"([\d.]+)B", model)
    return (model.split("-")[0], float(m.group(1)) if m else 0.0)


def fmt(xs):
    return " | ".join("–" if np.isnan(x) else f"{x:.1f}" for x in xs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--out-dir", default=os.path.join(here, "outputs"))
    p.add_argument("--split", default="valid_unseen")
    p.add_argument("--md", default=os.path.join(here, "RESULTS.md"))
    args = p.parse_args()

    runs = load_runs(args.out_dir, args.split)
    total = {"valid_unseen": 134, "valid_seen": 140}.get(args.split)

    paper = ["| Method | " + " | ".join(TYPES) + " | **Avg.** |", "|---|" + "---:|" * (len(TYPES) + 1)]
    diag = ["| Model | Episodes | All (%) | Invalid-action rate (%) | Mean ep. length | Model latency (ms/step) |",
            "|---|---:|---:|---:|---:|---:|"]

    STYLE_LABEL = {"strict": "Base Model, zero-shot (ours)", "rich": "Base Model, zero-shot + rich prompt (ours)"}
    last_base = None
    for key in sorted(runs, key=lambda k: (size_key(k.split("+")[0]), k)):
        path, eps = runs[key]
        model, _, style = key.partition("+")
        style = style or "strict"
        by_type = [100 * np.mean([e["success"] for e in eps if e["task_type"] == t])
                   if any(e["task_type"] == t for e in eps) else np.nan for t in TYPES]
        avg = np.nanmean(by_type)
        decisions = sum(e["steps"] for e in eps)
        partial = f" (partial {len(eps)}/{total})" if total and len(eps) < total else ""

        if model != last_base:
            paper.append(f"| ***{model.split('/')[-1]}*** |" + " |" * (len(TYPES) + 1))
            if model in REFERENCE:
                paper.append(f"| Base Model (paper) | {fmt(REFERENCE[model][:-1])} | **{REFERENCE[model][-1]:.1f}** |")
            last_base = model
        paper.append(f"| {STYLE_LABEL.get(style, style)}{partial} | {fmt(by_type)} | **{avg:.1f}** |")

        diag.append(
            f"| {model} ({style}) | {len(eps)} | {100 * np.mean([e['success'] for e in eps]):.1f} | "
            f"{100 * sum(e['invalid_outputs'] for e in eps) / max(decisions, 1):.1f} | "
            f"{np.mean([e['steps'] for e in eps]):.1f} | "
            f"{np.mean([s['latency_ms'] for e in eps for s in e['trajectory']]):.0f} |"
        )

    md = f"""# ALFWorld Zero-Shot Results

Success rate (%) on ALFWorld `{args.split}` (text-only `AlfredTWEnv`), by task type.

{chr(10).join(paper)}

**Avg.** is the unweighted mean of the 6 task-type success rates, as in the reference table.

## Diagnostics

{chr(10).join(diag)}

**All** is success over all episodes, weighted by how many games each task type has. **Invalid-action rate** is the share of decisions where the model's output couldn't be matched to an admissible command (the agent then played `look`).

## Protocol

- **Env:** official ALFWorld `AlfredTWEnv`, `{args.split}` split ({total} games, each played once), max 50 steps per episode.
- **Policy:** zero-shot. No training, no demonstrations, no expert plans.
- **Prompt:** task + current observation + last 5 (action, observation) pairs + numbered admissible commands. The model is asked to return only the exact action text.
- **Rich prompt** (`--prompt-style rich`): the same, plus a system prompt with generic ALFWorld rules (heat → microwave, cool → fridge, clean → sinkbasin, look → use desklamp, open closed receptacles, one object at a time, don't repeat "Nothing happens"), a step counter and the list of actions that had no effect. It is still zero-shot (no example trajectories, no per-game info), but it injects domain knowledge, which the strict setting doesn't.
- **Decoding:** greedy (`do_sample=False`), `max_new_tokens=32`, bf16 on one RTX 3090. Qwen3 thinking mode off (`enable_thinking=False`).
- **Latency:** the Qwen3 runs shared one GPU concurrently, so ms/step is inflated and only roughly comparable between models. Success numbers are unaffected.
- **Parsing:** exact match → cleaned (quotes/prefixes/numbering) → index → unique substring → unambiguous near-match; otherwise fallback `look`, counted as invalid.

The reference numbers come from a different protocol. That paper's prompt and decoding (e.g. reasoning before acting, history length, sampling) may differ, so compare trends rather than exact values.

Generated by `python report.py` from the trajectory files in `outputs/`.
"""
    with open(args.md, "w") as f:
        f.write(md)
    print(md)
    print(f"Wrote {args.md}")


if __name__ == "__main__":
    main()
