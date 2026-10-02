"""Render one logged ALFWorld episode as a terminal-style animated GIF.

ALFWorld (AlfredTWEnv) is text-only, so each frame shows the task, the step,
the LLM's chosen action and the environment's response.

  python make_demo_gif.py                                   # auto-pick a good successful episode
  python make_demo_gif.py --jsonl outputs/<run>.jsonl --episode 118
"""

import argparse
import glob
import json
import os
import textwrap

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
W, H, PAD, COLS = 900, 560, 22, 78
BG, FG, DIM, ACT, OK, BAD = "#1e1e2e", "#cdd6f4", "#7f849c", "#89b4fa", "#a6e3a1", "#f38ba8"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def auto_pick(out_dir):
    """Best complete run (highest success), then its shortest clean multi-stage success."""
    best = None
    for path in glob.glob(os.path.join(out_dir, "*.jsonl")):
        eps = load(path)
        if len(eps) < 134 or "task_type" not in eps[0]:
            continue
        rate = sum(e["success"] for e in eps) / len(eps)
        if best is None or rate > best[0]:
            best = (rate, path, eps)
    _, path, eps = best
    wins = [e for e in eps if e["success"]]
    pref = [e for e in wins if e["invalid_outputs"] == 0 and e["task_type"] not in ("Look", "Pick")] or wins
    return path, min(pref, key=lambda e: e["steps"])


def wrap(text, width=COLS):
    return [l for para in text.split("\n") for l in (textwrap.wrap(para, width) or [""])]


def frame(model, ep, upto, font, bold):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = PAD
    d.text((PAD, y), f"ALFWorld zero-shot | {model} | {ep['task_type']}", font=bold, fill=DIM)
    y += 26
    d.text((PAD, y), f"Task: {ep['task']}", font=bold, fill=FG)
    y += 34

    # Show the most recent steps that fit on screen.
    blocks = []
    for s in ep["trajectory"][: upto + 1]:
        lines = [(f"[{s['step'] + 1:2d}] > {s['action']}", ACT)]
        lines += [(f"     {l}", FG) for l in wrap(s["observation"], COLS - 5)]
        blocks.append(lines)
    footer = 2 if upto == len(ep["trajectory"]) - 1 else 0
    room = (H - y - PAD) // 19 - footer
    shown = []
    for b in reversed(blocks):
        if len(shown) + len(b) > room:
            break
        shown = b + shown
    for text, color in shown:
        d.text((PAD, y), text, font=font, fill=color)
        y += 19
    if footer:
        y += 10
        msg = f"SUCCESS in {len(ep['trajectory'])} steps" if ep["success"] else "FAILED"
        d.text((PAD, y), msg, font=bold, fill=OK if ep["success"] else BAD)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl")
    p.add_argument("--episode", type=int)
    p.add_argument("--out", default=os.path.join(HERE, "assets", "demo.gif"))
    args = p.parse_args()

    if args.jsonl:
        ep = next(e for e in load(args.jsonl) if e["episode"] == args.episode)
        path = args.jsonl
    else:
        path, ep = auto_pick(os.path.join(HERE, "outputs"))
    model = os.path.basename(path).split("_valid")[0].replace("_", "/", 1)

    font = ImageFont.truetype(FONT_PATH, 14)
    bold = ImageFont.truetype(FONT_PATH.replace(".ttf", "-Bold.ttf"), 15)
    frames = [frame(model, ep, i, font, bold) for i in range(len(ep["trajectory"]))]
    durations = [1400] * (len(frames) - 1) + [4000]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"{model} episode {ep['episode']} ({ep['task_type']}, {ep['steps']} steps, success={ep['success']}) -> {args.out}")


if __name__ == "__main__":
    main()
