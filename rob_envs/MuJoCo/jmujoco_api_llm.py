"""
Closed-loop multi-step cube sorting with a Franka Panda in MuJoCo, planned by a VLM:
  - GPT family via the OpenAI Responses API (gpt-5.5, gpt-5.4, gpt-5.4-mini, gpt-5.4-nano), or
  - Qwen3-VL-8B locally (same prompt), as a baseline.

Each step the VLM sees the current camera image, locates the cubes/mats (2D points -> 3D via depth),
and returns the remaining plan; the robot executes only the first action, then the loop re-observes
and re-plans until the VLM says the instruction is satisfied. Low-level motion is the scripted
differential-IK controller from jmujoco_jev.py.

Usage (from repo root, isaac_venv):
    export OPENAI_API_KEY=...
    # one episode
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_api_llm.py run --model gpt-5.4-mini --task 0 --seed 0
    # benchmark several models over all tasks, then plot
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_api_llm.py sweep \
        --models gpt-5.4-nano gpt-5.4-mini gpt-5.4 gpt-5.5 qwen --seeds 0 1
    isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_api_llm.py plot
Cameras: each step the VLM gets ONE image with two views side by side: an eye-to-hand side view of the whole
robot (left) and an eye-in-hand wrist camera beside the fingers (right). A point's half picks the camera whose
depth unprojects it to 3D.
Flags: --views both|scene|wrist   --perception gt (simulator poses; the VLM only plans)   --effort low|medium|high
Outputs: outputs/mujoco_llm/<model>[_gt][_<views>]/seed<N>_task<K>/{episode.mp4 (both views side by side),
         step_<k>.png (the image the VLM saw), final_<cam>.png, result.json},
         outputs/mujoco_llm/results.csv, outputs/mujoco_llm/comparison.png
"""

import argparse
import base64
import csv
import difflib
import glob
import io
import json
import os
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jmujoco_jev as J  # noqa: E402  (controller, rendering, depth unprojection, Qwen wrapper)

OUT_DIR = os.path.join(J.REPO_ROOT, "outputs/mujoco_llm")
J.IMG_W, J.IMG_H = 960, 540  # 16:9 like the MuJoCo viewer; J.render / J.pixel_to_world read these at call time

CUBES = {
    "red cube": [0.9, 0.1, 0.1, 1], "orange cube": [1.0, 0.5, 0.05, 1], "yellow cube": [0.95, 0.85, 0.1, 1],
    "green cube": [0.1, 0.75, 0.2, 1], "blue cube": [0.15, 0.35, 0.95, 1],
}
MAT_HALF = 0.085
MATS = {"purple mat": ([0.55, 0.28], [0.55, 0.2, 0.75, 1]), "white mat": ([0.55, -0.28], [0.95, 0.95, 0.95, 1])}
# Placement slots on a mat (2 x 3). Neighbors along x stay 8 cm apart, since the fingers open along x.
MAT_SLOTS = [(-0.04, -0.055), (-0.04, 0.0), (-0.04, 0.055), (0.04, -0.055), (0.04, 0.0), (0.04, 0.055)]
CUBE_SLOTS = [[0.40, 0.0], [0.52, 0.0], [0.64, 0.0], [0.40, 0.13], [0.40, -0.13],
              [0.56, 0.12], [0.56, -0.12], [0.70, 0.06]]
MAX_STEPS = 8
MEMORY_M = 3  # past (state image, action) pairs shown to the VLM each step

# Cameras: eye-to-hand side view ("front") and eye-in-hand wrist camera ("wrist") beside the fingers
WRIST_POS, WRIST_FOVY = np.array([0.06, 0.0, 0.03]), 75
VIEWS = {"both": ["front", "wrist"], "scene": ["front"], "wrist": ["wrist"]}
VIEW_DESC = {
    "front": "the side camera viewing the whole robot and the table (eye-to-hand); the purple mat is behind the "
             "cubes and the white mat in front of them",
    "wrist": "the camera mounted on the robot's gripper, looking down along the fingers at the table (eye-in-hand); "
             "the two fingertips are visible at the top edge",
}

WARM, COOL = ["red cube", "orange cube", "yellow cube"], ["green cube", "blue cube"]
# (instruction, expected final mat per cube; None = must not end up on a mat)
TASKS = [
    ("put every warm-colored cube on the purple mat and every cool-colored cube on the white mat",
     {**{c: "purple mat" for c in WARM}, **{c: "white mat" for c in COOL}}),
    ("move all cubes to the white mat except the blue one",
     {c: (None if c == "blue cube" else "white mat") for c in CUBES}),
    ("put the cubes whose colors appear in a traffic light on the purple mat, leave the others where they are",
     {c: ("purple mat" if c in ("red cube", "yellow cube", "green cube") else None) for c in CUBES}),
    ("sort the cubes: primary colors on the white mat, all other colors on the purple mat",
     {c: ("white mat" if c in ("red cube", "yellow cube", "blue cube") else "purple mat") for c in CUBES}),
    ("put the cube that is the color of the sky on the purple mat and the cube the color of a pumpkin on the white mat",
     {c: {"blue cube": "purple mat", "orange cube": "white mat"}.get(c) for c in CUBES}),
]

# USD per 1M tokens (input, output) for cost tracking; update if pricing changes.
PRICES = {"gpt-5.5": (5.0, 30.0), "gpt-5.4": (2.5, 15.0), "gpt-5.4-mini": (0.75, 4.5), "gpt-5.4-nano": (0.2, 1.25)}


# ============================================================
# Scene
# ============================================================

def local_lookat_quat(pos, target, up):
    """Camera quat (looks along -z, y up) in its parent body's frame."""
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([r, np.cross(r, f), -f]).flatten())
    return quat


def build_scene(rng):
    spec = mujoco.MjSpec.from_file(J.PANDA_XML)
    spec.delete(spec.key("home"))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 0, 0])
    # Wrist camera: offset from the hand, perpendicular to the finger axis, looking just past the fingertips
    spec.body("hand").add_camera(name="wrist", pos=WRIST_POS, fovy=WRIST_FOVY,
                                 quat=local_lookat_quat(WRIST_POS, [0, 0, 0.2], [-1, 0, 0]))  # rolled 180 deg
    for body in spec.bodies:
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1.0
    for name, (xy, rgba) in MATS.items():
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*xy, 0.001],
                                size=[MAT_HALF, MAT_HALF, 0.001], rgba=rgba, contype=0, conaffinity=0)
    slots = rng.permutation(CUBE_SLOTS)
    for (name, rgba), xy in zip(CUBES.items(), slots):
        body = spec.worldbody.add_body(name=name.replace(" ", "_"), pos=[*(xy + rng.uniform(-0.015, 0.015, 2)), J.CUBE_HALF])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[J.CUBE_HALF] * 3, rgba=rgba,
                      mass=0.05, friction=[1.5, 0.01, 0.001], condim=4)
    # Same view as the MuJoCo viewer's default camera for the Panda scene (azimuth 120, elevation -20, fovy 45),
    # re-centered a little toward the workspace so the whole robot, the cubes and both mats are in frame.
    lookat, az, el, dist = np.array([0.30, 0.0, 0.30]), np.deg2rad(120), np.deg2rad(-20), 1.75
    cam_pos = lookat - dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    spec.worldbody.add_camera(name="front", pos=cam_pos, quat=J.lookat_quat(cam_pos, lookat), fovy=45)

    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = J.IMG_W, J.IMG_H
    # Softer lighting: the default overexposes the cube tops, so orange reads as yellow and blue as cyan
    model.vis.headlight.diffuse[:], model.vis.headlight.ambient[:] = 0.3, 0.25
    model.light_diffuse[:], model.light_specular[:], model.mat_specular[:] = 0.5, 0.1, 0.1
    data = mujoco.MjData(model)
    data.qpos[:7], data.qpos[7:9] = J.HOME_Q, 0.04
    data.ctrl[:7], data.ctrl[7] = J.HOME_Q, 255
    mujoco.mj_forward(model, data)
    return model, data


def render_cam(model, data, renderer, cam, depth=False):
    if depth:
        renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    img = renderer.render().copy()
    renderer.disable_depth_rendering()
    return img


class DualViewController(J.Controller):
    """J.Controller whose video shows the scene and wrist views side by side."""

    def __init__(self, model, data, renderer, writer):
        super().__init__(model, data, renderer, None)
        self.video = writer

    def step(self, target_pos):
        super().step(target_pos)
        if self.steps % self.render_every == 0:
            self.video.append_data(np.concatenate(
                [render_cam(self.m, self.d, self.renderer, c) for c in VIEWS["both"]], axis=1))


def cube_pos(data, name):
    return data.body(name.replace(" ", "_")).xpos.copy()


def mat_of(xyz):
    for mat, (xy, _) in MATS.items():
        if np.all(np.abs(xyz[:2] - xy) < MAT_HALF) and xyz[2] < 0.05:
            return mat
    return None


# ============================================================
# Planners (same prompt for GPT and Qwen)
# ============================================================

def build_prompt(task, history, perception, cams, memory=()):
    done_txt = "\n".join(f"{i + 1}. attempted: pick {a['pick']}, place on {a['place']}"
                         for i, a in enumerate(history)) or "none yet"
    mem_txt = (f"\nMemory: before the current image you also get the last {len(memory)} previous state images, oldest "
               "first, each labeled with the action the robot attempted from that state. Compare each state with the "
               "next one (and the last with the current image) to check whether the action actually worked: a cube "
               "that did not move means the grasp missed, and a cube that moved but is not on the mat was misplaced. "
               "Re-plan from what you see in the CURRENT image; retry failed actions if needed.\n" if memory else "")
    multi = len(cams) > 1
    halves = ["LEFT half", "RIGHT half"]
    images_txt = ("You get ONE image with the two camera views of the current scene side by side:\n" +
                  "\n".join(f"- {halves[i]}: {VIEW_DESC[c]}." for i, c in enumerate(cams)) if multi else
                  f"You get one camera image of the current scene: {VIEW_DESC[cams[0]]}.")
    locate = ('"objects": [{"label": "<color> cube" or "purple mat" or "white mat", "point_2d": [x, y]}, ...], '
              if perception == "llm" else "")
    locate_rule = (("First locate every cube and both mats. " + (
                    "Give each object ONCE, in the half where you see it best: cubes preferably in the RIGHT half "
                    "(the gripper camera looking down on the table) when visible there, mats in the LEFT half where "
                    "they are fully visible. point_2d is the object's "
                    "center in coordinates normalized to 0-1000 over the WHOLE combined image (x from its left edge "
                    "to its right edge, y down), so points in the right half have x > 500. " if multi else
                    "point_2d is the object's center in coordinates normalized to 0-1000 (x to the right, y down). "))
                   if perception == "llm" else f"The cubes in the scene are: {', '.join(CUBES)}. ")
    return (
        "You control a robot arm that sorts small cubes onto two flat mats, called \"purple mat\" and \"white mat\". "
        f"{images_txt}\n"
        f"Instruction: {task}\n"
        f"Actions attempted so far:\n{done_txt}\n{mem_txt}\n"
        f"{locate_rule}"
        "Then decide which cubes still have to be moved to satisfy the instruction, judging from the images which "
        "cubes already sit on which mat. Never move a cube that is already on its correct mat, and never move a cube "
        "the instruction says to leave alone. Each action picks one cube and places it on one mat.\n"
        "Answer ONLY with JSON:\n"
        f'{{{locate}"remaining_plan": [{{"pick": "<color> cube", "place": "purple mat" or "white mat"}}, ...], '
        '"done": true if the instruction is already satisfied else false}'
    )


def png_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class OpenAIPlanner:
    def __init__(self, model, effort=None, detail="high"):
        from openai import OpenAI
        self.client, self.model, self.effort, self.detail = OpenAI(), model, effort, detail
        self.name = model

    def __call__(self, img, prompt, memory=()):
        kwargs = {"reasoning": {"effort": self.effort}} if self.effort else {}
        content = [{"type": "input_text", "text": prompt}]
        for label, past in memory_items(memory, img):
            content += [{"type": "input_text", "text": label},
                        {"type": "input_image", "image_url": f"data:image/png;base64,{png_b64(past)}",
                         "detail": self.detail}]
        resp = self.client.responses.create(model=self.model, input=[{"role": "user", "content": content}], **kwargs)
        u = resp.usage
        p_in, p_out = PRICES.get(self.model, (0.0, 0.0))
        cost = (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6
        return resp.output_text, {"in_tokens": u.input_tokens, "out_tokens": u.output_tokens, "cost_usd": cost}


class QwenPlanner:
    name = "qwen"

    def __init__(self, vlm=None):
        self.vlm = vlm or J.VLM()

    def __call__(self, img, prompt, memory=()):
        content = [{"type": "text", "text": prompt}]
        for label, past in memory_items(memory, img):
            content += [{"type": "text", "text": label}, {"type": "image", "image": Image.fromarray(past)}]
        proc, model = self.vlm.processor, self.vlm.model
        inputs = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                          add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt").to(model.device)
        with self.vlm.torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        text = proc.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text, {"in_tokens": int(inputs["input_ids"].shape[1]), "out_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
                      "cost_usd": 0.0}


def memory_items(memory, current):
    """(label, image) pairs: the past states with the action attempted from each, then the current state."""
    items = [(f"Previous state (step {m['step']}), from which the robot attempted: pick {m['action']['pick']}, "
              f"place on {m['action']['place']}:", m["image"]) for m in memory]
    return items + [("CURRENT state:" if memory else "Current state:", current)]


def make_planner(name, effort=None, vlm=None):
    return QwenPlanner(vlm) if name == "qwen" else OpenAIPlanner(name, effort)


# ============================================================
# Episode: observe -> plan -> execute first action -> repeat
# ============================================================

def snap(label, options):
    return (difflib.get_close_matches(str(label).lower().strip(), options, 1, 0.6) or [None])[0]


def series_tag(model_name, perception, views, memory_m=MEMORY_M):
    return (model_name + ("_gt" if perception == "gt" else "") + ("" if views == "both" else f"_{views}")
            + ("" if memory_m == MEMORY_M else f"_mem{memory_m}"))


def run_episode(planner, task_id, seed, perception="llm", views="both", out_root=OUT_DIR, memory_m=MEMORY_M):
    task, expected = TASKS[task_id]
    cams = VIEWS[views]
    out_dir = os.path.join(out_root, series_tag(planner.name, perception, views, memory_m), f"seed{seed}_task{task_id}")
    os.makedirs(out_dir, exist_ok=True)

    model, data = build_scene(np.random.default_rng(seed))
    renderer = mujoco.Renderer(model, height=J.IMG_H, width=J.IMG_W)
    for _ in range(200):
        mujoco.mj_step(model, data)
    mat_fill = {m: 0 for m in MATS}
    history, steps_log = [], []
    t_episode = time.perf_counter()
    writer = imageio.get_writer(os.path.join(out_dir, "episode.mp4"), fps=30)
    ctrl = DualViewController(model, data, renderer, writer)
    home_tcp = data.site(ctrl.site).xpos.copy()
    memory = []  # (state image, attempted action) pairs; the VLM sees the last memory_m of them

    for step in range(MAX_STEPS):
        # One image for the VLM: the camera views side by side (scene left, wrist right)
        combined = np.concatenate([render_cam(model, data, renderer, c) for c in cams], axis=1)
        depths = [render_cam(model, data, renderer, c, depth=True) for c in cams]
        Image.fromarray(combined).save(os.path.join(out_dir, f"step_{step}.png"))

        t0 = time.perf_counter()
        try:
            recent = memory[-memory_m:] if memory_m > 0 else []
            raw, usage = planner(combined, build_prompt(task, history, perception, cams, recent), recent)
        except Exception as e:  # API errors end the episode but are logged
            steps_log.append({"step": step, "error": f"{type(e).__name__}: {e}"})
            print(f"[step {step}] planner error: {e}")
            break
        latency = time.perf_counter() - t0
        out = J.parse_json(raw, "{}") or {}
        log = {"step": step, "latency_s": round(latency, 2), **usage, "done": out.get("done"),
               "plan": out.get("remaining_plan"), "raw": raw if not out else None}
        steps_log.append(log)

        plan = out.get("remaining_plan") or []
        if out.get("done") or not plan:
            print(f"[step {step}] planner says done ({latency:.1f}s)")
            break

        # Resolve the first action to 3D targets
        pick, place = snap(plan[0].get("pick"), list(CUBES)), snap(plan[0].get("place"), list(MATS))
        if perception == "gt":
            pick_xyz = cube_pos(data, pick) if pick else None
            mat_xy = np.array(MATS[place][0]) if place else None
        else:
            # Points are normalized over the combined image: the half they fall in picks the camera whose depth
            # unprojects them
            points, source = {}, {}
            for o in out.get("objects") or []:
                try:
                    x, y = (float(v) for v in o["point_2d"])
                    label = snap(o["label"], list(CUBES) + list(MATS))
                except (KeyError, TypeError, ValueError):
                    continue
                px = x / 1000 * J.IMG_W * len(cams)
                idx = min(int(px // J.IMG_W), len(cams) - 1)
                u = px - idx * J.IMG_W
                # Cubes: prefer the top-down wrist view. Mats: prefer the side view, where they are fully visible.
                preferred = "front" if label in MATS else "wrist"
                if label is None or (source.get(label) == preferred and cams[idx] != preferred):
                    continue
                points[label] = J.pixel_to_world(model, data, model.camera(cams[idx]).id,
                                                 u, y / 1000 * J.IMG_H, depths[idx])
                source[label] = cams[idx]
            log["point_source"] = source
            pick_xyz = points.get(pick)
            mat_xy = points[place][:2] if place in points else None
        log.update(action={"pick": pick, "place": place})
        if pick is None or place is None or pick_xyz is None or mat_xy is None:
            log["invalid_action"] = True
            print(f"[step {step}] could not ground action {plan[0]}")
            history.append({"pick": f"(failed to locate) {plan[0].get('pick')}", "place": plan[0].get("place")})
            memory.append({"step": step, "image": combined, "action": history[-1]})
            continue

        slot = MAT_SLOTS[mat_fill[place] % len(MAT_SLOTS)]
        mat_fill[place] += 1
        before = cube_pos(data, pick)
        # Diagnostics only (never shown to the VLM): where it aimed vs. where things really are
        log.update(pick_xyz=np.round(pick_xyz, 3).tolist(), mat_xy=np.round(mat_xy, 3).tolist(),
                   pick_err_cm=round(float(np.linalg.norm(pick_xyz[:2] - before[:2]) * 100), 1),
                   mat_err_cm=round(float(np.linalg.norm(np.asarray(mat_xy) - MATS[place][0]) * 100), 1))
        ctrl.pick_and_place(pick_xyz, np.array([mat_xy[0] + slot[0], mat_xy[1] + slot[1], 0.0]))
        ctrl.move_to(home_tcp)  # clear the camera's view before the next observation
        landed = mat_of(cube_pos(data, pick))
        log.update(landed_on=landed, moved_cm=round(float(np.linalg.norm(cube_pos(data, pick) - before) * 100), 1))
        history.append({"pick": pick, "place": place})
        memory.append({"step": step, "image": combined, "action": history[-1]})
        print(f"[step {step}] {pick} -> {place} | landed on {landed} | aim err: cube {log['pick_err_cm']} cm "
              f"({log.get('point_source', {}).get(pick, 'sim')}), mat {log['mat_err_cm']} cm | "
              f"plan {len(plan)} left | {latency:.1f}s")

    writer.close()
    for c in VIEWS["both"]:
        Image.fromarray(render_cam(model, data, renderer, c)).save(os.path.join(out_dir, f"final_{c}.png"))
    renderer.close()

    final_mats = {c: mat_of(cube_pos(data, c)) for c in CUBES}
    correct = {c: final_mats[c] == expected[c] for c in CUBES}
    result = {
        "model": planner.name, "perception": perception, "views": views, "memory_m": memory_m, "task_id": task_id, "task": task, "seed": seed,
        "success": all(correct.values()), "cube_accuracy": round(float(np.mean(list(correct.values()))), 3),
        "final_mats": final_mats, "expected": expected,
        "n_actions": len([s for s in steps_log if "action" in s]), "n_planner_calls": len(steps_log),
        "planner_said_done": bool(steps_log and steps_log[-1].get("done")),
        "invalid_actions": sum(bool(s.get("invalid_action")) for s in steps_log),
        "grasp_misses": sum(1 for s in steps_log if "landed_on" in s and s["landed_on"] != s["action"]["place"]),
        "errors": [s["error"] for s in steps_log if "error" in s],
        "planner_latency_s": [s["latency_s"] for s in steps_log if "latency_s" in s],
        "cost_usd": round(sum(s.get("cost_usd", 0) for s in steps_log), 5),
        "tokens": [sum(s.get("in_tokens", 0) for s in steps_log), sum(s.get("out_tokens", 0) for s in steps_log)],
        "episode_s": round(time.perf_counter() - t_episode, 1), "steps": steps_log,
        "out_dir": os.path.relpath(out_dir, J.REPO_ROOT),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[result] success={result['success']} cube_acc={result['cube_accuracy']} "
          f"actions={result['n_actions']} cost=${result['cost_usd']:.4f} -> {out_dir}")
    return result


# ============================================================
# Sweep + plot
# ============================================================

def load_results():
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_DIR, "*", "*", "result.json")))]


def key(r):
    return r["model"], r["perception"], r.get("views", "scene"), r.get("memory_m", 0)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from compare_modes import AXIS, GRID, INK, INK_2, MUTED, SURFACE, style

    # Fixed categorical order (dataviz reference palette slots 1-6), assigned by series, never cycled
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    order = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "qwen"]
    series = sorted({key(r) for r in rows},
                    key=lambda s: (order.index(s[0]) if s[0] in order else len(order), s[1], s[2], s[3]))
    label = lambda s: (s[0] + (" (sim poses)" if s[1] == "gt" else "") + ("" if s[2] == "both" else f" ({s[2]} only)")
                       + ("" if s[3] == MEMORY_M else f" (memory {s[3]})"))
    color = {s: palette[i % len(palette)] for i, s in enumerate(series)}
    by = {s: [r for r in rows if key(r) == s] for s in series}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    fig.suptitle("Multi-step cube sorting: GPT family vs Qwen3-VL", x=0.03, ha="left", fontsize=14,
                 color=INK, fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color[s]) for s in series]
    fig.legend(handles, [f"{label(s)} — {len(by[s])} episodes" for s in series], loc="upper left",
               bbox_to_anchor=(0.025, 0.955), ncol=min(3, len(series)), frameon=False, fontsize=8.5,
               labelcolor=INK_2, handlelength=1.2)

    def bars(ax, values, fmt, title, ylabel, log=False):
        x = np.arange(len(series))
        for i, s in enumerate(series):
            v = values[s]
            if v is None:
                ax.text(i, 0, "n/a", ha="center", va="bottom", fontsize=8, color=MUTED)
                continue
            ax.bar(i, v, 0.6, color=color[s], edgecolor=SURFACE, linewidth=2)
            ax.text(i, v * (1.08 if log else 1) + (0 if log else 1.5), fmt(v), ha="center", va="bottom",
                    fontsize=8, color=INK_2)
        if log:
            ax.set_yscale("log")
        ax.set_xticks(x, [label(s).replace(" (", "\n(") for s in series], fontsize=8)
        style(ax, title, ylabel)

    mean = lambda s, f: float(np.mean([f(r) for r in by[s]])) if by[s] else None
    bars(axes[0, 0], {s: 100 * mean(s, lambda r: r["success"]) for s in series}, lambda v: f"{v:.0f}%",
         "Task success (all 5 cubes correct)", "% of episodes")
    axes[0, 0].set_ylim(0, 112)
    bars(axes[0, 1], {s: 100 * mean(s, lambda r: r["cube_accuracy"]) for s in series}, lambda v: f"{v:.0f}%",
         "Per-cube accuracy", "% of cubes on the right mat")
    axes[0, 1].set_ylim(0, 112)
    lat = {s: (float(np.median([l for r in by[s] for l in r["planner_latency_s"]]))
               if any(r["planner_latency_s"] for r in by[s]) else None) for s in series}
    bars(axes[1, 0], lat, lambda v: f"{v:.1f}s", "Planner latency per step (median, log scale)", "seconds", log=True)
    cost = {s: (mean(s, lambda r: r["cost_usd"]) if s[0] != "qwen" else None) for s in series}
    bars(axes[1, 1], cost, lambda v: f"${v:.3f}", "API cost per episode (mean)", "USD")
    axes[1, 1].text(0.99, 0.97, "qwen runs locally (no API cost)", transform=axes[1, 1].transAxes, ha="right",
                    va="top", fontsize=8, color=MUTED)

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = os.path.join(OUT_DIR, "comparison.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def write_csv(rows):
    path = os.path.join(OUT_DIR, "results.csv")
    keys = ["model", "perception", "views", "memory_m", "task_id", "seed", "success", "cube_accuracy", "n_actions", "n_planner_calls",
            "invalid_actions", "grasp_misses", "cost_usd", "episode_s", "out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def summarize(rows):
    print(f"\n{'model':22s} {'n':>3s} {'success%':>9s} {'cube_acc%':>10s} {'actions':>8s} {'$/ep':>8s}")
    for k in sorted({key(r) for r in rows}):
        rs = [r for r in rows if key(r) == k]
        print(f"{series_tag(*k):22s} {len(rs):3d} {100 * np.mean([r['success'] for r in rs]):9.0f} "
              f"{100 * np.mean([r['cube_accuracy'] for r in rs]):10.0f} {np.mean([r['n_actions'] for r in rs]):8.1f} "
              f"{np.mean([r['cost_usd'] for r in rs]):8.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one episode")
    r.add_argument("--model", default="gpt-5.4-mini", help="gpt-5.5 | gpt-5.4 | gpt-5.4-mini | gpt-5.4-nano | qwen")
    r.add_argument("--task", type=int, default=0, help=f"task index 0-{len(TASKS) - 1}")
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sweep", help="benchmark models x tasks x seeds, then plot")
    s.add_argument("--models", nargs="+", default=["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "qwen"])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    s.add_argument("--tasks", nargs="+", type=int, default=list(range(len(TASKS))))
    for p in (r, s):
        p.add_argument("--perception", choices=["llm", "gt"], default="llm",
                       help="llm: the VLM localizes objects; gt: simulator poses, the VLM only plans")
        p.add_argument("--memory", type=int, default=MEMORY_M,
                       help="past (state image, action) pairs given to the VLM each step (0 = no memory)")
        p.add_argument("--views", choices=list(VIEWS), default="both",
                       help="camera images given to the model: both (eye-to-hand + eye-in-hand), scene or wrist")
        p.add_argument("--effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                       help="GPT reasoning effort (default: model default)")
    sub.add_parser("plot", help="re-plot saved results")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    models = [args.model] if args.cmd == "run" else args.models if args.cmd == "sweep" else []
    if any(m != "qwen" for m in models) and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Run `export OPENAI_API_KEY=...` first.")

    if args.cmd == "run":
        run_episode(make_planner(args.model, args.effort), args.task, args.seed, args.perception, args.views,
                    memory_m=args.memory)
        return
    if args.cmd == "sweep":
        vlm = J.VLM() if "qwen" in args.models else None
        for m in args.models:
            planner = make_planner(m, args.effort, vlm)
            for seed in args.seeds:
                for t in args.tasks:
                    print(f"\n===== {m} | seed {seed} | task {t}: {TASKS[t][0]}")
                    try:
                        run_episode(planner, t, seed, args.perception, args.views, memory_m=args.memory)
                    except Exception as e:
                        print(f"[ERROR] {type(e).__name__}: {e}")
    rows = load_results()
    if not rows:
        sys.exit(f"No results under {OUT_DIR}")
    summarize(rows)
    print(f"\nCSV  -> {write_csv(rows)}\nPlot -> {plot(rows)}")


if __name__ == "__main__":
    main()
