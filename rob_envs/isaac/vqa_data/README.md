# VQA / VLA data collection (Isaac Lab, Franka)

Generates **(image, language, answer)** data where the answer is either a **robot action** (for VLA / policy
training) or a **text answer** about the scene (for VQA / perception training). Labels are exact because they come
from the simulator.

## What happens
1. **Scene**: Franka Panda, 3 cubes (red / green / blue) at random non-overlapping positions and yaws, one fixed RGB camera (224×224).
2. **Instruction**: random color + template, e.g. *"pick up the red cube"*, *"lift the blue block"*.
3. **Scripted expert** (differential IK, 20 Hz): rotate the gripper top-down and align it to the cube (not recorded), then
   approach above → descend → close gripper → lift. The end-effector setpoint moves ≤ 1 cm per step.
4. **Recording**: every step saves the camera frame plus the action that the expert took from that frame.
5. **Filter**: only episodes where the cube ends up > 10 cm above the table are kept (failed attempts are retried).

## Run (from repo root)
```bash
python rob_envs/isaac/vqa_data/collect_franka_vqa.py --episodes 50              # headless
python rob_envs/isaac/vqa_data/collect_franka_vqa.py --episodes 5 --viz kit     # watch in GUI
python rob_envs/isaac/vqa_data/collect_franka_vqa.py --episodes 2 --debug       # per-phase logs + GIF per attempt
```
Flags: `--out` (default `outputs/isaac/vqa_franka`), `--img_size` (224), `--seed`, `--max_tries` (default 3×episodes).

## Output
```
outputs/isaac/vqa_franka/
  images/ep0000_t000.png ...   frames
  actions.jsonl                one line per step   (VLA data)
  qa.jsonl                     ~10 Q&A per episode (VQA data, from the first frame)
  meta.json                    action format, binning, camera, counts
```

**`actions.jsonl`**: image + instruction → action
```json
{"episode": 0, "step": 39, "image": "images/ep0000_t039.png", "instruction": "grasp the blue cube and lift it",
 "action": [0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0], "action_tokens": "128 128 64 128 128 128 255", "is_last": false}
```
- `action` = `[dx, dy, dz, droll, dpitch, dyaw, gripper]`: end-effector position change in m per step (robot base frame);
  rotation deltas are 0 (fixed top-down grasp); gripper `+1` open, `-1` close.
- `action_tokens` = each dim binned into 256 bins (RT-2 / OpenVLA style), so the action can be a **text answer**:
  position over ±0.02 m, rotation over ±π, gripper over ±1. `128` ≈ zero.

**`qa.jsonl`**: image + question → text answer (ground truth from sim)
```json
{"episode": 0, "image": "images/ep0000_t000.png", "question": "Where is the red cube in the image? Answer in pixel coordinates (x, y).", "answer": "(155, 104)"}
```
Question types: cube pixel location, cube 3D position, closest cube to gripper, which cube is on the left, cube count,
is the gripper holding an object.

## Using it
- **VLA / policy**: prompt = `instruction`, input = `image`, target = `action_tokens` (as text) or `action` (continuous head).
- **VQA fine-tune**: prompt = `question`, input = `image`, target = `answer`.
- Mix both to teach scene understanding and control together.

## Extending
- More tasks: add a phase list in `run_episode` (e.g. place on another cube, push).
- More variation: colors, cube sizes, lighting, camera pose, distractor objects.
- Wrist camera: add a second `Camera` attached to `/World/Franka/panda_hand`.
