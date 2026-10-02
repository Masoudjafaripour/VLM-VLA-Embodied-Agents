"""Run the zero-shot LLM agent in ALFWorld's embodied twin (AlfredThorEnv / AI2-THOR)
on one game and save the egocentric robot view as a GIF.

Same text interface as the AlfredTWEnv eval (task, observation, admissible
commands -> one command); high-level commands are executed in the 3D scene by
ALFWorld's oracle controller. Needs `ai2thor==2.1.0` and an X display:

  DISPLAY=:0 python thor_demo.py --model Qwen/Qwen3-1.7B \
      --game ~/.cache/alfworld/json_2.1.1/valid_unseen/<task>/<trial>
"""

import argparse
import json
import os
import random
import re
from os.path import join as pjoin

import numpy as np
from alfworld.agents.environment import get_environment
from PIL import Image, ImageDraw, ImageFont

from eval_zero_shot import extract_task, make_config
from llm_agent import LLMAgent

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def caption(frame_bgr, lines, size):
    img = Image.fromarray(frame_bgr[:, :, ::-1].copy()).resize((size, size))
    font = ImageFont.truetype(FONT, 15)
    bar = 24 * len(lines) + 12
    canvas = Image.new("RGB", (size, size + bar), "#1e1e2e")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    for i, (text, color) in enumerate(lines):
        d.text((10, size + 8 + 24 * i), text[:64], font=font, fill=color)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--game", required=True, help="trial folder containing traj_data.json")
    p.add_argument("--prompt-style", default="strict", choices=["strict", "rich"])
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--history-len", type=int, default=5)
    p.add_argument("--size", type=int, default=400)
    p.add_argument("--out", default=pjoin(HERE, "assets", "thor_demo.gif"))
    args = p.parse_args()
    random.seed(0)

    cfg = make_config(args.max_steps)
    cfg["env"]["thor"] = {"screen_width": 300, "screen_height": 300, "smooth_nav": False,
                          "save_frames_to_disk": False, "save_frames_path": "/tmp/alfworld_frames"}
    cfg["controller"] = {"type": "oracle", "debug": False, "load_receps": True}

    env = get_environment("AlfredThorEnv")(cfg, train_eval="eval_out_of_distribution")
    env.get_env_paths = lambda: None  # keep only our game (init_env re-scans the split otherwise)
    env.json_file_list = [pjoin(os.path.expanduser(args.game), "traj_data.json")]
    env = env.init_env(batch_size=1)

    agent = LLMAgent(args.model)
    obs, info = env.reset()
    obs = obs[0]
    task = extract_task(obs)
    print("Task:", task)

    frames = [caption(env.get_frames()[0], [(f"Task: {task}", "#cdd6f4"), ("[start]", "#7f849c")], args.size)]
    history, failed, won, log = [], [], False, []
    for t in range(args.max_steps):
        admissible = info["admissible_commands"][0]
        action, meta = agent.act(task, obs, history[-args.history_len:], admissible,
                                 style=args.prompt_style, step=t, max_steps=args.max_steps, failed=failed[-10:])
        # alfworld 0.4.2 bug: the THOR controller offers "move OBJ to RECEP" as admissible, but its
        # parser (agents/controller/base.py) only executes the older "put OBJ in/on RECEP" form.
        obs, _, done, info = env.step([re.sub(r"^move (.+) to (.+)$", r"put \1 in/on \2", action)])
        obs, done, won = obs[0], bool(done[0]), bool(info["won"][0])
        history.append((action, obs))
        if obs.strip() == "Nothing happens." and action not in failed:
            failed.append(action)
        log.append({"step": t, "action": action, "observation": obs, **meta})
        print(f"[{t + 1:2d}] {action}  ->  {obs[:90]}")
        frames.append(caption(env.get_frames()[0], [(f"Task: {task}", "#cdd6f4"),
                                                    (f"[{t + 1:2d}] > {action}", "#89b4fa")], args.size))
        if done or won:
            break

    end = ("SUCCESS" if won else "FAILED") + f" in {len(log)} steps"
    frames.append(caption(env.get_frames()[0], [(f"Task: {task}", "#cdd6f4"),
                                                (end, "#a6e3a1" if won else "#f38ba8")], args.size))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=[1200] * (len(frames) - 1) + [3500], loop=0)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"model": args.model, "game": args.game, "task": task, "success": won, "trajectory": log}, f, indent=1)
    print(f"{end} -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
