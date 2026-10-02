# VLM-VLA-Embodied-Agents

A research playground for **language- and vision-driven robot control**: robot manipulation where a vision-language model (VLM) either answers questions about the scene (**VQA**) or outputs actions directly (**VLA**).

The repo builds this up in layers, from low-level control through learning to large models:
**controllers** (differential IK, null-space control) → **planners** (an LLM turns an instruction into subgoals) → **learned policies** (DAgger, PPO) → **VLAs** evaluated on standard benchmarks → **simulated VQA/VLA data** for training them.

<img src="assets/panda.png" width="400">

## Core idea

The answer to a visual question can be a fact about the scene or the next action:

```python
img = camera.capture()

answer = vlm.ask(img, "Where is the red cube?")             # VQA: scene understanding -> structured state
action = vla.get_action(img, "pick up the red cube")        # VLA: answer = action (e.g. 7-D EE delta + gripper)

robot.execute(action)                                       # repeat until the task is done
```

## What's in the repo

| Area | Where | What |
|---|---|---|
| Classical control (MuJoCo) | [src/](src/), [src/llm_vlm_robot_control/](src/llm_vlm_robot_control/) | Franka pick-and-place with waypoints, UR5 differential IK with null-space obstacle avoidance (based on [mjctrl](https://github.com/kevinzakka/mjctrl)) |
| LLM planning | [src/llm_vlm_robot_control/llm_control.py](src/llm_vlm_robot_control/llm_control.py) | Qwen2.5-3B reads the instruction and scene state, returns JSON subgoals, and diff-IK executes them in MuJoCo |
| Imitation learning | [src/BC/](src/BC/) | DAgger on 2D/3D reaching arms with state, vision and fused policies |
| Benchmarks | [baseline/LIBERO/](baseline/LIBERO/), [baseline/CALVIN/](baseline/CALVIN/) | Working env setups plus a pluggable `get_action(image, instruction)` rollout loop |
| LLM agents | [baseline/ALFWorld/alfworld_eval/](baseline/ALFWorld/alfworld_eval/) | Zero-shot Qwen3 evaluation on ALFWorld (text env + AI2-THOR embodied demo) |
| VLA evaluation | [baseline/Models/](baseline/Models/) | One `Policy` interface for OpenVLA, SmolVLA, pi0/pi0.5, RT-1, Octo, RT-2, CoT-VLA |
| Isaac Lab | [rob_envs/isaac/](rob_envs/isaac/) | Franka and H1 humanoid sims, PPO walking training, and scripted **VQA/VLA dataset collection** |
| VLM + decision model (MuJoCo) | [rob_envs/MuJoCo/](rob_envs/MuJoCo/) | Franka pick-and-place where Qwen3-VL perceives and **Jev** (TypeSafe AI) makes typed decisions; qwen / jev / combined modes benchmarked |


## UR5 Null Space Control
<img src="assets/ur5_nullspace_control.png" width="400">

Implements differential inverse kinematics with null space control on a UR5e arm in MuJoCo. The end-effector tracks a waypoint trajectory while avoiding a spherical obstacle, with joint velocities solved via a damped pseudoinverse Jacobian.

## Isaac Lab Simulation

Minimal [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 3.0 / Isaac Sim 6.1 examples in a separate `isaac_venv` (Python 3.12). See [rob_envs/isaac/](rob_envs/isaac/) for install and commands.

- **Franka arm**: scripted sinusoidal joint motion, with an optional GIF and joint-angle plot (`franka_sim.py --record`).
- **H1 humanoid walking**: NVIDIA's pretrained RSL-RL policy (`h1_walk.sh`), or train PPO from scratch with live learning-curve plots (`h1_train.sh`).
- **VQA / VLA data collection**: a scripted IK expert picks up a named cube and records `(image, instruction, action)` per step, with actions also binned to 256 text tokens. It also saves ground-truth `(image, question, answer)` pairs about the scene. See [rob_envs/isaac/vqa_data/](rob_envs/isaac/vqa_data/).

<img src="assets/isaac/franka_arm.gif" width="240"> <img src="assets/isaac/h1_walk.gif" width="320">

**VQA sample**: start → grasp → lift, instruction *"grasp the blue cube and lift it"*

<img src="assets/isaac/vqa_sample.png" width="680">

| Question / instruction | Answer |
|---|---|
| Where is the blue cube in the image? (pixel x, y) | `(88, 110)` |
| What is the 3D position of the red cube in meters? | `(0.36, 0.18, 0.02)` |
| Which cube is closest to the gripper? | `blue` |
| Which cube is on the left side of the image? | `blue` |
| *grasp the blue cube and lift it* (step 0 action) | `[0.0006, -0.0042, -0.0090, 0, 0, 0, +1]` → tokens `131 101 70 128 128 128 255` |
| *grasp the blue cube and lift it* (step 47 action) | `[0, 0, 0, 0, 0, 0, -1]` (close gripper) → tokens `128 128 128 128 128 128 0` |

## Jev × Qwen3-VL Pick-and-Place (MuJoCo)

A Franka Panda follows instructions such as *"the grass-colored block belongs on the white mat"*.
[Jev](https://pydantic.dev/docs/ai/models/typesafe/) is TypeSafe AI's text-only *System 1* decision model, and here it makes the decisions:
- **Pick and place:** given the instruction and the list of detected objects, Jev chooses which cube and which mat. The answer is a typed choice restricted to objects in the scene, with a confidence for each field.
- **Success check:** Jev gives a yes/no on whether the task succeeded.

[Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) handles perception (2D points on the camera image, lifted to 3D with depth). In `combined` mode Qwen also acts as *System 2*, reasoning step by step only when Jev's confidence is below 0.7. A differential-IK controller executes the grasp.

Over 50 episodes per mode, Jev chose the right cube and mat every time, including paraphrases, negations and Qwen's habit of calling the purple mat "pink". It decided in a median of 0.17–0.25 s and never had to escalate to Qwen. The remaining task failures come from Qwen localizing a cube several cm off. See [rob_envs/MuJoCo/](rob_envs/MuJoCo/) for the three modes, commands and the full comparison.

<img src="rob_envs/MuJoCo/assets/demo_jev.gif" width="320"> <img src="rob_envs/MuJoCo/assets/comparison.png" width="480">

## LIBERO Benchmark

Baseline setup for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark, with a pluggable VLM/VLA rollout loop. See [baseline/LIBERO/](baseline/LIBERO/) for setup, scripts, and details.

<img src="baseline/LIBERO/outputs/libero_spatial_task0.png" width="300"> <img src="baseline/LIBERO/outputs/libero_spatial_task0_rollout.gif" width="300">

## CALVIN Benchmark

Baseline setup for the [CALVIN](https://github.com/mees/calvin) long-horizon, language-conditioned manipulation benchmark, with the same pluggable VLM/VLA rollout loop. See [baseline/CALVIN/](baseline/CALVIN/) for setup, scripts, and details.

<img src="baseline/CALVIN/outputs/calvin_scene_D_lift_pink_block_table.png" width="300"> <img src="baseline/CALVIN/outputs/calvin_scene_D_lift_red_block_table_rollout.gif" width="300">

## ALFWorld Benchmark (zero-shot LLM agents)

Zero-shot evaluation of open-source LLMs (Qwen3-0.6B/1.7B/4B) on [ALFWorld](https://github.com/alfworld/alfworld) household tasks, with no training, no demonstrations and no expert plans. At each step the LLM picks one admissible command from the task, observation and recent history. See [baseline/ALFWorld/alfworld_eval/](baseline/ALFWorld/alfworld_eval/) for setup and scripts, and [RESULTS.md](baseline/ALFWorld/alfworld_eval/RESULTS.md) for the full per-task-type table.

**Embodied (AI2-THOR) demo**: the same zero-shot agent (Qwen3-4B) acting in `AlfredThorEnv`, ALFWorld's 3D AI2-THOR twin of the text games, shown from the robot's egocentric camera (left). Next to it is a text-env episode (Qwen3-1.7B) heating an egg and placing it (right).

<img src="baseline/ALFWorld/alfworld_eval/assets/thor_demo.gif" width="300"> <img src="baseline/ALFWorld/alfworld_eval/assets/demo.gif" width="480">

Best result: Qwen3-1.7B with a richer zero-shot prompt reaches **14.8** avg. success on `valid_unseen` (strict prompt: 5.0; a reference paper reports 12.4 for the same base model).

## VLA Evaluation

A shared `Policy` interface plus wrappers for OpenVLA, SmolVLA, pi0/pi0.5, RT-1, Octo, RT-2, and CoT-VLA, evaluated on the LIBERO and CALVIN baselines above. OpenVLA (real 7B, LIBERO-finetuned checkpoint), SmolVLA, and RT-1 (a stateful `tf_agents` checkpoint with real Universal Sentence Encoder instruction embeddings) are actually validated with real weights, each in its own venv (`vla_venv`/`vla_venv_lerobot`/`vla_venv_tf` - different, conflicting framework/torch-version requirements); the rest are wired up with correct loading code but not executed here (a fourth framework, multi-GB checkpoints, or no public weights at all). See [baseline/Models/](baseline/Models/) for the eval harness, per-model status, and sample results.
