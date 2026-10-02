# ALFWorld Zero-Shot LLM Baseline

Zero-shot evaluation of an open-source Hugging Face instruct LLM on [ALFWorld](https://github.com/alfworld/alfworld) text-only tasks (`AlfredTWEnv`). No training, no fine-tuning, no few-shot demonstrations, no expert/oracle plans:

$$\pi_\theta(a_t \mid g, o_t, h_t, \mathcal{A}_t)$$

At every step the LLM sees the task $g$, the current observation $o_t$, the last few (action, observation) pairs $h_t$ and the admissible commands $\mathcal{A}_t$. It must return one of those commands verbatim.

## Files

| File | What it does |
|---|---|
| [`eval_zero_shot.py`](eval_zero_shot.py) | Builds the official `AlfredTWEnv` and runs the episode loop. Logs each episode to JSONL and prints/saves summary metrics. |
| [`llm_agent.py`](llm_agent.py) | Loads the HF model (bf16 on GPU) and runs greedy generation. Maps the raw output to an admissible command with `parse_action`. |
| [`prompts.py`](prompts.py) | The zero-shot prompt (system message + task/observation/history/numbered actions). |
| [`report.py`](report.py) | Collects all `*_summary.json` runs into a paper-style table: success % per task type (Pick/Look/Clean/Heat/Cool/Pick2), Avg., All. |
| [`requirements.txt`](requirements.txt) | Python deps. |

## Setup

This uses its own `alfworld_venv/` at the repo root, not `vla_venv/`. Qwen3 needs `transformers>=4.51`, but `vla_venv` pins `transformers==4.40.1` for the OpenVLA baselines.

```bash
python3 -m venv alfworld_venv
alfworld_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
alfworld_venv/bin/pip install "transformers>=4.51" accelerate alfworld numpy   # text-only alfworld; no AI2-THOR
alfworld_venv/bin/alfworld-download   # game files + PDDL/grammar -> ~/.cache/alfworld (~2.3 GB)
source alfworld_venv/bin/activate
```

Set `ALFWORLD_DATA` to use a different data location.

## Running

```bash
cd baseline/ALFWorld/alfworld_eval
python eval_zero_shot.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --num-episodes 20 \
  --max-steps 50 \
  --seed 42
```

Other flags:
- `--split {valid_unseen,valid_seen,train}` (default `valid_unseen`, the standard 134-game out-of-distribution eval set).
- `--history-len` sets how many recent steps go into the prompt (default 5).
- `--max-new-tokens` (default 32).
- `--out-dir` sets where results go (default `./outputs`).

Decoding is deterministic: `do_sample=False, temperature=None, top_p=None, top_k=None`. The model's `generation_config` sampling defaults are overridden. Qwen3's thinking mode is turned off (`enable_thinking=False` in the chat template), so the model answers with the action directly.

### Qwen3 size sweep (paper-style table)

```bash
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B; do
  python eval_zero_shot.py --model $m --num-episodes -1 --max-steps 50 --seed 42
done
python report.py
```

`--num-episodes -1` plays every game in the split exactly once (134 for `valid_unseen`). Decoding is greedy and history resets every episode, so on the full split a different `--seed` only changes the order of games, not the results. One run per model is enough.

`report.py` prints per-type success %, plus two averages:
- **Avg.**: the unweighted mean of the 6 type rates. This is how the reference tables report it. For Qwen3-1.7B, (6.8+60.2+0+0+3.6+4.1)/6 = 12.4.
- **All**: success over all episodes, weighted by how many games each type has.

## Episode selection

`AlfredTWEnv` collects games in `os.walk` order, which depends on the filesystem. The script sorts that list and then calls the env's own `env.seed(seed)`, which shuffles the game order. So `--seed` fully determines which `--num-episodes` games are played, and two models run with the same seed see the same games.

## Output parsing

`parse_action(raw, admissible)` tries these in order and records which one fired as `parse_mode`:

1. `exact`: raw output is an admissible command.
2. `cleaned`: same after taking the first line, stripping quotes, markdown, `Action:`-style prefixes, list numbering and a trailing period, and ignoring case and whitespace.
3. `index`: a bare number pointing into the numbered action list.
4. `substring`: exactly one admissible command appears as a whole phrase inside the output.
5. `fuzzy`: the closest match, accepted only when unambiguous. Either equal ignoring spaces (`cabinet1` → `cabinet 1`) or the unique completion of a truncated command (`... with microwave` → `... with microwave 1`). Character-similarity scores aren't used, because they rate `cabinet 1` vs `cabinet 2` about as close as a real typo.
6. `fallback`: otherwise the agent plays `look` (or the first admissible command). This counts as an **invalid-output event**.

The LLM can never send a command outside `admissible_commands` to the env.

## Outputs

Each run writes two files to `outputs/`:

- `<model>_<split>_s<seed>_<timestamp>.jsonl` has one line per episode:
  ```json
  {"episode": 0, "task": "...", "task_type": "Clean", "gamefile": "...", "success": true, "reward": 1.0, "steps": 17,
   "invalid_outputs": 1, "latency_ms_mean": 42.3,
   "trajectory": [{"step": 0, "action": "...", "observation": "...", "raw_output": "...",
                   "parse_mode": "exact", "latency_ms": 40.1}, ...]}
  ```
- `<...>_summary.json` holds the summary metrics:
  - `success_rate`
  - `mean_reward`
  - `mean_episode_length`
  - `mean_latency_ms_per_decision`
  - `invalid_action_rate` (fallbacks ÷ total decisions)
  - `success_rate_by_type` / `episodes_by_type`

Latency is wall-clock `model.generate` time per decision, measured with CUDA sync.

## Notes

- Reward is ALFWorld's sparse score: 1.0 when the goal is achieved, 0 otherwise.
- The ALFWorld `max_episode_steps` limit is set to `--max-steps` too, so env termination and the loop bound match.
- The task string is parsed from the initial observation (`"Your task is to: ..."`). Nothing else from the game file (`traj_data.json`, PDDL, expert plan) is shown to the model.
