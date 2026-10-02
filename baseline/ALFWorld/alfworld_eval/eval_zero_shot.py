"""Zero-shot ALFWorld (text-only AlfredTWEnv) evaluation of an HF instruct LLM.

No training, no demonstrations, no expert plans: at each step the LLM sees the
task, current observation, recent history and the admissible commands, and
must return one admissible command.
"""

import argparse
import json
import os
import random
import time
from os.path import join as pjoin

import numpy as np
import torch
from alfworld.agents.environment import get_environment
from alfworld.info import ALFWORLD_DATA

from llm_agent import LLMAgent

SPLITS = {
    "train": "train",
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def make_config(max_steps):
    """Minimal subset of ALFWorld's base_config.yaml that AlfredTWEnv reads."""
    data = pjoin(ALFWORLD_DATA, "json_2.1.1")
    return {
        "env": {
            "type": "AlfredTWEnv",
            "goal_desc_human_anns_prob": 0.0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",  # unused: expert wrapper is only added for train+dagger
        },
        "dataset": {
            "data_path": pjoin(data, "train"),
            "eval_id_data_path": pjoin(data, "valid_seen"),
            "eval_ood_data_path": pjoin(data, "valid_unseen"),
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": pjoin(ALFWORLD_DATA, "logic", "alfred.pddl"),
            "grammar": pjoin(ALFWORLD_DATA, "logic", "alfred.twl2"),
        },
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": max_steps}},
    }


# Short names for ALFWorld's 6 task types (as used in paper tables).
TASK_TYPE_NAMES = {
    "pick_and_place_simple": "Pick",
    "look_at_obj_in_light": "Look",
    "pick_clean_then_place_in_recep": "Clean",
    "pick_heat_then_place_in_recep": "Heat",
    "pick_cool_then_place_in_recep": "Cool",
    "pick_two_obj_and_place": "Pick2",
}


def task_type_of(gamefile):
    # .../valid_unseen/<task_type>-<obj>-<...>/trial_xxx/game.tw-pddl
    folder = os.path.basename(os.path.dirname(os.path.dirname(gamefile)))
    return TASK_TYPE_NAMES.get(folder.split("-")[0], "Unknown")


def extract_task(obs):
    marker = "Your task is to: "
    return obs.split(marker, 1)[1].strip() if marker in obs else obs.strip()


def run_episode(env, agent, max_steps, history_len, prompt_style="strict"):
    obs, info = env.reset()
    obs = obs[0]
    gamefile = info["extra.gamefile"][0]
    task = extract_task(obs)

    history, trajectory, failed = [], [], []
    won, done, reward = False, False, 0.0
    for t in range(max_steps):
        admissible = info["admissible_commands"][0]
        action, meta = agent.act(task, obs, history[-history_len:] if history_len else [], admissible,
                                 style=prompt_style, step=t, max_steps=max_steps, failed=failed[-10:])

        obs, score, done, info = env.step([action])
        obs, reward, done, won = obs[0], float(score[0]), bool(done[0]), bool(info["won"][0])

        trajectory.append({"step": t, "action": action, "observation": obs, **meta})
        history.append((action, obs))
        if obs.strip() == "Nothing happens." and action not in failed:
            failed.append(action)
        if done or won:
            break

    lat = [s["latency_ms"] for s in trajectory]
    return {
        "task": task,
        "task_type": task_type_of(gamefile),
        "gamefile": gamefile,
        "success": won,
        "reward": reward,
        "steps": len(trajectory),
        "invalid_outputs": sum(s["parse_mode"] == "fallback" for s in trajectory),
        "latency_ms_mean": float(np.mean(lat)) if lat else 0.0,
        "trajectory": trajectory,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--num-episodes", type=int, default=20, help="-1 = every game in the split once (134 for valid_unseen)")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="valid_unseen", choices=list(SPLITS))
    p.add_argument("--history-len", type=int, default=5, help="number of recent (action, obs) pairs in the prompt")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prompt-style", default="strict", choices=["strict", "rich"],
                   help="rich = add generic ALFWorld rules + progress hints (still zero-shot)")
    p.add_argument("--out-dir", default=pjoin(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Official ALFWorld text env. Sort game files so --seed fully determines the episode order
    # (AlfredTWEnv collects them in os.walk order, which is filesystem-dependent).
    tw_env = get_environment("AlfredTWEnv")(make_config(args.max_steps), train_eval=SPLITS[args.split])
    tw_env.game_files = sorted(tw_env.game_files)
    env = tw_env.init_env(batch_size=1)
    env.seed(args.seed)
    if args.num_episodes < 0:
        args.num_episodes = tw_env.num_games  # shuffled_cycle visits each game once per pass

    agent = LLMAgent(args.model, max_new_tokens=args.max_new_tokens)
    print(f"Model {args.model} on {agent.device} ({agent.dtype})")

    tag = args.model.replace("/", "_") + ("" if args.prompt_style == "strict" else f"+{args.prompt_style}")
    run_name = f"{tag}_{args.split}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(args.out_dir, exist_ok=True)
    traj_path = pjoin(args.out_dir, run_name + ".jsonl")

    results = []
    with open(traj_path, "w") as f:
        for ep in range(args.num_episodes):
            r = {"episode": ep, **run_episode(env, agent, args.max_steps, args.history_len, args.prompt_style)}
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[ep {ep:3d}] success={r['success']!s:5} steps={r['steps']:3d} "
                  f"invalid={r['invalid_outputs']:2d} lat={r['latency_ms_mean']:.1f}ms | {r['task']}")
    env.close()

    n_decisions = sum(r["steps"] for r in results)
    summary = {
        "model": args.model,
        "prompt_style": args.prompt_style,
        "split": args.split,
        "seed": args.seed,
        "num_episodes": len(results),
        "max_steps": args.max_steps,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_episode_length": float(np.mean([r["steps"] for r in results])),
        "mean_latency_ms_per_decision": float(np.mean([s["latency_ms"] for r in results for s in r["trajectory"]])),
        "invalid_action_rate": sum(r["invalid_outputs"] for r in results) / max(n_decisions, 1),
        "success_rate_by_type": {
            t: float(np.mean([r["success"] for r in results if r["task_type"] == t]))
            for t in TASK_TYPE_NAMES.values() if any(r["task_type"] == t for r in results)
        },
        "episodes_by_type": {
            t: sum(r["task_type"] == t for r in results)
            for t in TASK_TYPE_NAMES.values() if any(r["task_type"] == t for r in results)
        },
        "trajectories": traj_path,
    }
    with open(pjoin(args.out_dir, run_name + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== Summary ====")
    for k, v in summary.items():
        if isinstance(v, dict):
            v = "  ".join(f"{t}={x:.3f}" if isinstance(x, float) else f"{t}={x}" for t, x in v.items())
        print(f"{k:30s} {v:.4f}" if isinstance(v, float) else f"{k:30s} {v}")


if __name__ == "__main__":
    main()
