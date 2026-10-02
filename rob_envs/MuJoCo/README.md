# MuJoCo pick-and-place: Qwen3-VL × Jev

A Franka Panda picks a colored cube and places it on a colored mat, following a natural-language instruction
(*"put the cube that is the color of a banana on the mat that is not white"*). This setup compares two
models in different roles:

- **[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**, a local vision-language model. It **sees**: from the
  camera image it returns a 2D point for each object, which is converted to 3D with the depth render.
- **[Jev](https://pydantic.dev/docs/ai/models/typesafe/)** (TypeSafe AI), a hosted *System 1* decision model. It **decides**: typed
  choices (which cube, which mat) and yes/no answers, each with a calibrated confidence. Jev is text-only.

Neither model moves the arm. A scripted controller does that: differential IK on the TCP following
approach → grasp → lift → place waypoints.

## The three modes

| Mode | Perception | Decision (pick / place) | Success self-check |
|---|---|---|---|
| `qwen` | Qwen3-VL | Qwen3-VL | Qwen3-VL looks at the final image |
| `jev` | simulator poses (no VLM) | Jev | n/a (no image description) |
| `combined` | Qwen3-VL | **Jev = System 1**, falling back to **Qwen = System 2** when Jev's confidence < 0.7 | Jev judges Qwen's description; Qwen reasons over the image if Jev is unsure |

In `combined`, Qwen's step-by-step System 2 reasoning runs only when Jev's per-field confidence
(`result.response.provider_details['confidence']`) is below `S1_CONF_THRESH`.

## Run (from repo root)

Setup: use `isaac_venv` (mujoco 3.11, transformers 5, `pydantic-ai-slim[typesafe]`), and fetch the Franka model:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie external/mujoco_menagerie
git -C external/mujoco_menagerie sparse-checkout set franka_emika_panda
export TYPESAFE_API_KEY=...        # needed for jev / combined
```

```bash
# one episode
MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_jev.py --mode combined --seed 0 \
    --task "put the cube that is the color of a banana on the mat that is not white"

# benchmark all three modes, 50 episodes each (6 instructions, cycled over seeds), then plot
MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/compare_modes.py --episodes 50
isaac_venv/bin/python rob_envs/MuJoCo/compare_modes.py --plot-only        # re-plot saved runs
```
Each episode writes `outputs/mujoco_jev/<mode>/seed<N>_<task>/` containing `pick_place.mp4`, `vlm_input.png`, `final.png` and
`result.json` (decisions, confidences, System 2 reasoning, timings). The sweep also writes `results.csv` and `comparison.png`.

## Demos (seed 0)

| `qwen` | `jev` | `combined` |
|---|---|---|
| ![qwen demo](assets/demo_qwen.gif) | ![jev demo](assets/demo_jev.gif) | ![combined demo](assets/demo_combined.gif) |
| *"put the cube that is the color of a banana on the mat that is not white"*: yellow → purple mat | *"the grass-colored block belongs on the white mat"*: green → white mat (Jev confidence 0.97) | *"move the green cube to the purple mat"*: Qwen called the mat "pink"; Jev still matched it (confidence 0.96) |

## Results (50 episodes per mode, real Jev API)

![comparison](assets/comparison.png)

| Mode | Decision correct | Task success | Self-check agrees w/ sim | Decision latency (median) |
|---|---|---|---|---|
| `qwen` | 100% | 88% | 98% | 0.42 s |
| `jev` | 100% | 100% | n/a | 0.25 s |
| `combined` | 100% | 88% | 74% | **0.17 s** |

- **Decisions are solved in every mode.** Jev handled the paraphrases, negations and the pink/purple naming mismatch at high
  confidence, so `combined` never escalated to System 2 (0/50) and made the fastest decisions.
- **Task failures come from perception, not decisions.** `qwen` and `combined` share the same misses: in a few layouts
  Qwen localizes a cube 6–8 cm off (median error 1.8 cm, against a 2 cm half-cube width), and the grasp misses. `jev` uses
  exact simulator poses, so it succeeds every time.
- **Jev's success self-check is too literal.** When Qwen says *"the cube is on the **pink** mat"* and the instruction
  said *purple*, Jev confidently answers "no" (confidence 0.74–0.96, above the threshold, so the check never escalates).
  That causes 13 false negatives. Possible fixes are to describe the scene using the instruction's own object names,
  or to raise the threshold for the judge.

## Files

| File | What it does |
|---|---|
| `jmujoco_jev.py` | Scene (built with `MjSpec`), Qwen perception, the Jev / Qwen / cascade `Decider`, IK controller, `run_episode()` |
| `compare_modes.py` | Benchmark sweep over modes × tasks × seeds, `results.csv`, `comparison.png` |
| `assets/` | Demo GIFs and the comparison plot used in this README |
