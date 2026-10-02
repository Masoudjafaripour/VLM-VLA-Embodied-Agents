"""
Pick-and-place with a Franka Panda in MuJoCo, driven by
  - Qwen3-VL-8B-Instruct  : perception (2D grounding on the camera image -> 3D via depth)
  - TypeSafe Jev          : typed decisions (which object, which target, did it succeed?)
  - Differential IK       : low-level execution of scripted grasp/place waypoints

Jev is a text-only decision model, so the VLM turns pixels into text/points and
Jev turns (instruction + scene description) into calibrated, schema-constrained choices.

Modes (--mode):
    qwen      Qwen3-VL perceives AND decides (no Jev)
    jev       simulator poses for perception, Jev decides (no VLM loaded)
    combined  Qwen3-VL perceives; Jev is System 1 (fast typed decision + confidence) and Qwen is
              System 2, reasoning step by step only when Jev's confidence < S1_CONF_THRESH

Usage (from repo root, isaac_venv has mujoco 3.11 + transformers 5 + pydantic-ai):
    export TYPESAFE_API_KEY=...            # without it, Jev falls back to keyword matching
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_jev.py --mode combined \
        --task "put the red cube on the purple mat" --seed 0
Each run saves to outputs/mujoco_jev/<mode>/seed<N>_<task>/ (video, images, result.json).
To compare the three modes over many tasks/seeds, use compare_modes.py.
"""

import argparse
import difflib
import json
import os
import re
import time
from typing import Literal

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from pydantic import Field, create_model

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANDA_XML = os.path.join(REPO_ROOT, "external/mujoco_menagerie/franka_emika_panda/scene.xml")
OUT_DIR = os.path.join(REPO_ROOT, "outputs/mujoco_jev")
VLM_NAME = "Qwen/Qwen3-VL-8B-Instruct"
JEV_MODEL = "typesafe:jev-latest"

CUBE_HALF = 0.02
CUBES = {"red cube": [0.9, 0.1, 0.1, 1], "green cube": [0.1, 0.8, 0.2, 1], "yellow cube": [0.95, 0.85, 0.1, 1]}
MATS = {"purple mat": ([0.55, 0.25], [0.55, 0.2, 0.75, 1]), "white mat": ([0.55, -0.25], [0.95, 0.95, 0.95, 1])}
HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
IMG_W, IMG_H = 640, 480

# mode -> (perception source, decision backend)
# "cascade" = Jev as fast System 1; when its confidence is below S1_CONF_THRESH, Qwen reasons as System 2.
MODES = {"qwen": ("qwen", "qwen"), "jev": ("gt", "jev"), "combined": ("qwen", "cascade")}
S1_CONF_THRESH = 0.7

# Differential IK gains (same scheme as mjctrl's diffik)
INTEGRATION_DT, DAMPING, MAX_STEP = 0.1, 1e-4, 0.008


# ============================================================
# Scene
# ============================================================

def lookat_quat(pos, target):
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([r, u, -f]).flatten())
    return quat


def build_scene(rng):
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    spec.delete(spec.key("home"))  # its qpos size no longer matches once cubes are added
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 0, 0])
    for body in spec.bodies:  # gravity compensation, otherwise the arm sags away from IK targets
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1.0

    for name, (xy, rgba) in MATS.items():
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*xy, 0.001],
                                size=[0.07, 0.07, 0.001], rgba=rgba, contype=0, conaffinity=0)

    # Cubes in a random, non-overlapping arrangement between the two mats
    slots = rng.permutation([[0.45, 0.0], [0.62, 0.0], [0.40, 0.12], [0.40, -0.12], [0.68, 0.10]])
    for (name, rgba), xy in zip(CUBES.items(), slots):
        xy = xy + rng.uniform(-0.02, 0.02, 2)
        body = spec.worldbody.add_body(name=name.replace(" ", "_"), pos=[*xy, CUBE_HALF])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3, rgba=rgba,
                      mass=0.05, friction=[1.5, 0.01, 0.001], condim=4)

    cam_pos = np.array([1.35, 0.0, 0.75])
    spec.worldbody.add_camera(name="front", pos=cam_pos, quat=lookat_quat(cam_pos, [0.5, 0, 0]), fovy=50)

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:7] = HOME_Q
    data.qpos[7:9] = 0.04
    data.ctrl[:7] = HOME_Q
    data.ctrl[7] = 255
    mujoco.mj_forward(model, data)
    return model, data


# ============================================================
# Perception: Qwen3-VL grounding -> 3D points via depth
# ============================================================

class VLM:
    def __init__(self, name=VLM_NAME):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()

    def ask(self, image, prompt, max_new_tokens=512):
        content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                                    add_generation_prompt=True, return_dict=True,
                                                    return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text, bracket="[]"):
    m = re.search(re.escape(bracket[0]) + r".*" + re.escape(bracket[1]), text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def pixel_to_world(model, data, cam_id, u, v, depth):
    """Unproject pixel (u, v) using the rendered depth (MuJoCo camera looks along -z, y up)."""
    f = 0.5 * IMG_H / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    d = depth[int(np.clip(v, 0, IMG_H - 1)), int(np.clip(u, 0, IMG_W - 1))]
    p_cam = np.array([(u - IMG_W / 2) / f * d, -(v - IMG_H / 2) / f * d, -d])
    return data.cam_xpos[cam_id] + data.cam_xmat[cam_id].reshape(3, 3) @ p_cam


def render(model, data, renderer, depth=False):
    if depth:
        renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="front")
    img = renderer.render().copy()
    renderer.disable_depth_rendering()
    return img


def perceive_vlm(vlm, model, data, renderer, out_dir):
    rgb = render(model, data, renderer)
    depth = render(model, data, renderer, depth=True)
    Image.fromarray(rgb).save(os.path.join(out_dir, "vlm_input.png"))

    prompt = ("Locate every small colored cube and every flat colored mat on the floor in front of the robot. "
              "Answer only with a JSON list with one entry per object, e.g. "
              '[{"label": "blue cube", "point_2d": [x, y]}, {"label": "black mat", "point_2d": [x, y]}]. '
              'Each label must be "<color> cube" or "<color> mat" using the object\'s own color, and point_2d is '
              "the center of the object in coordinates normalized to 0-1000.")
    raw = vlm.ask(Image.fromarray(rgb), prompt)
    print(f"[VLM] raw grounding:\n{raw}\n")

    cam_id = model.camera("front").id
    scene = {}
    for det in parse_json(raw) or []:
        try:
            x, y = det["point_2d"]
            label = det.get("description") or det["label"]  # some generations put the color in "description"
        except (KeyError, TypeError, ValueError):
            continue
        u, v = x / 1000 * IMG_W, y / 1000 * IMG_H
        scene[label.lower().strip()] = pixel_to_world(model, data, cam_id, u, v, depth)
    return scene


def perceive_gt(model, data):
    scene = {name: data.body(name.replace(" ", "_")).xpos.copy() for name in CUBES}
    scene.update({name: np.array([*xy, 0.0]) for name, (xy, _) in MATS.items()})
    return scene


def canonical(label, scene, gt):
    """Map a (possibly VLM-invented) scene label to the true entity nearest to its estimated position."""
    if label is None or label not in scene:
        return None
    return min(gt, key=lambda k: np.linalg.norm(scene[label][:2] - gt[k][:2]))


# ============================================================
# Decisions: TypeSafe Jev (typed choices / yes-no), or Qwen, or keyword fallback
# ============================================================

class Decider:
    def __init__(self, backend, vlm=None):
        has_key = bool(os.environ.get("TYPESAFE_API_KEY"))
        if backend == "jev" and not has_key:
            print("[Jev] TYPESAFE_API_KEY not set -> falling back to keyword matching")
            backend = "keyword"
        if backend == "cascade" and not has_key:
            print("[Jev] TYPESAFE_API_KEY not set -> System 1 falls back to keyword matching")
            backend = "keyword+qwen"
        self.backend, self.vlm = backend, vlm
        self.info = {}  # System 1 / System 2 trace of the last call, for result.json

    @staticmethod
    def split(scene):
        objects = [k for k in scene if "cube" in k] or list(scene)
        targets = [k for k in scene if k not in objects] or list(scene)
        return objects, targets

    # ---- System 1 -------------------------------------------------------
    @staticmethod
    def _jev(output_type, instructions, prompt):
        """Run Jev; returns (output, lowest per-field confidence)."""
        from pydantic_ai import Agent
        result = Agent(JEV_MODEL, output_type=output_type, instructions=instructions).run_sync(prompt)
        conf = (result.response.provider_details or {}).get("confidence") or {}
        return result.output, min(conf.values()) if conf else 0.0

    @staticmethod
    def _keyword_s1(task, objects, targets):
        """Keyword stand-in for Jev when there is no API key: confident only if a unique option is named."""
        def best(options):
            scores = {o: sum(w in task.lower() for w in o.split()) for o in options}
            top = max(scores.values())
            winners = [o for o, s in scores.items() if s == top]
            return winners[0], float(len(winners) == 1 and top == len(winners[0].split()))
        (pick, c1), (place, c2) = best(objects), best(targets)
        return pick, place, min(c1, c2)

    # ---- System 2 -------------------------------------------------------
    def _qwen_reason_pick_place(self, task, scene_txt, objects, targets, s1_guess):
        raw = self.vlm.ask(None, f"Instruction: {task}\nVisible objects:\n{scene_txt}\n\n"
                                 f"A fast system guessed pick={s1_guess[0]!r}, place={s1_guess[1]!r} but was unsure. "
                                 "The object names come from a vision model and may name colors loosely (e.g. pink "
                                 "for purple): map the instruction to the closest matching object. Reason briefly, "
                                 "step by step, about what the instruction refers to (colors, paraphrases, negations), "
                                 "then on the last line answer only with JSON "
                                 f'{{"pick": one of {objects}, "place": one of {targets}}}.', 768)
        out = parse_json(raw.strip().splitlines()[-1] if raw.strip() else "", "{}") or parse_json(raw, "{}") or {}
        # Unparseable reasoning (e.g. cut off) keeps System 1's answer rather than an arbitrary option
        snap = lambda v, opts, s1: (difflib.get_close_matches(str(v), opts, 1, 0) or [s1])[0] if v else s1
        return snap(out.get("pick"), objects, s1_guess[0]), snap(out.get("place"), targets, s1_guess[1]), raw

    def _qwen_reason_success(self, task, final_img, description):
        raw = self.vlm.ask(Image.fromarray(final_img),
                           f"Instruction given to the robot: {task}\nA first look said: {description}\n"
                           "Look carefully at the image and reason step by step about where the relevant cube is "
                           "and whether it lies on the right mat. End with a final line 'ANSWER: yes' or "
                           "'ANSWER: no'.", 256)
        m = re.findall(r"answer:\s*(yes|no)", raw.lower())
        return (m[-1] == "yes") if m else None, raw

    def choose_pick_place(self, task, scene):
        objects, targets = self.split(scene)
        scene_txt = "\n".join(f"- {k} at x={p[0]:.2f}, y={p[1]:.2f}" for k, p in scene.items())

        if self.backend == "jev":
            PickPlace = create_model(
                "PickPlace",
                pick=(Literal[tuple(objects)], Field(description="The object the instruction asks the robot to move")),
                place=(Literal[tuple(targets)], Field(description="Where the instruction asks the object to be put")),
            )
            out, conf = self._jev(PickPlace, "You ground a robot instruction to the objects visible in the scene.",
                                  f"Instruction: {task}\nVisible objects:\n{scene_txt}")
            self.info = {"s1_confidence": conf}
            return out.pick, out.place

        if self.backend in ("cascade", "keyword+qwen"):
            t0 = time.perf_counter()
            if self.backend == "cascade":
                PickPlace = create_model(
                    "PickPlace",
                    pick=(Literal[tuple(objects)], Field(description="The object the instruction asks to move")),
                    place=(Literal[tuple(targets)], Field(description="Where the instruction asks to put it")),
                )
                out, conf = self._jev(PickPlace, "You ground a robot instruction to the objects visible in the scene.",
                                      f"Instruction: {task}\nVisible objects:\n{scene_txt}")
                pick, place = out.pick, out.place
            else:
                pick, place, conf = self._keyword_s1(task, objects, targets)
            self.info = {"s1_pick": pick, "s1_place": place, "s1_confidence": round(conf, 3),
                         "s1_time_s": round(time.perf_counter() - t0, 3), "escalated": conf < S1_CONF_THRESH}
            print(f"[System 1] pick='{pick}', place='{place}', confidence={conf:.2f}")
            if conf < S1_CONF_THRESH:
                pick, place, reasoning = self._qwen_reason_pick_place(task, scene_txt, objects, targets, (pick, place))
                self.info["s2_reasoning"] = reasoning
                print(f"[System 2] escalated (conf < {S1_CONF_THRESH}) -> pick='{pick}', place='{place}'")
            return pick, place

        if self.backend == "qwen":
            raw = self.vlm.ask(None, f"Instruction: {task}\nVisible objects:\n{scene_txt}\n\n"
                                     f"Which object should the robot pick (one of {objects}) and where should it "
                                     f"place it (one of {targets})? Answer only with JSON "
                                     '{"pick": "...", "place": "..."}.', 64)
            out = parse_json(raw, "{}") or {}
            snap = lambda v, opts: (difflib.get_close_matches(str(v), opts, 1, 0) or [opts[0]])[0]
            return snap(out.get("pick"), objects), snap(out.get("place"), targets)

        def best(options):
            return max(options, key=lambda o: sum(w in task.lower() for w in o.split()))
        return best(objects), best(targets)

    def judge_success(self, task, final_img, description):
        """Returns the backend's verdict, or None when it can't judge (no VLM description / keyword fallback)."""
        if self.backend == "qwen":
            ans = self.vlm.ask(Image.fromarray(final_img), f"Instruction given to the robot: {task}\n"
                                                           "Was the instruction completed? Answer yes or no.", 8)
            return ans.strip().lower().startswith("yes")
        judge_instr = ("Given an instruction and a description of the scene after the robot acted, "
                       "decide whether the instruction was completed.")
        judge_prompt = f"Instruction: {task}\nScene after acting: {description}"
        if self.backend in ("cascade", "keyword+qwen") and description is not None:
            if self.backend == "cascade":
                verdict, conf = self._jev(bool, judge_instr, judge_prompt)
            else:
                verdict, conf = None, 0.0  # keyword matching can't judge a scene description
            self.info = {"s1_verdict": verdict, "s1_confidence": round(conf, 3), "escalated": conf < S1_CONF_THRESH}
            print(f"[System 1] success={verdict}, confidence={conf:.2f}")
            if conf < S1_CONF_THRESH:
                verdict, reasoning = self._qwen_reason_success(task, final_img, description)
                self.info["s2_reasoning"] = reasoning
                print(f"[System 2] escalated -> success={verdict}")
            return verdict
        if self.backend == "jev" and description is not None:
            verdict, _ = self._jev(bool, judge_instr, judge_prompt)
            return verdict
        return None


# ============================================================
# Control: differential IK on the TCP site + scripted waypoints
# ============================================================

class Controller:
    def __init__(self, model, data, renderer, writer):
        self.m, self.d, self.renderer, self.writer = model, data, renderer, writer
        self.site = model.site("tcp").id
        self.dof = np.arange(7)
        self.target_quat = np.zeros(4)
        down = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)  # z down, matches the home orientation
        mujoco.mju_mat2Quat(self.target_quat, down.flatten())
        self.render_every = max(1, int(1 / 30 / model.opt.timestep))
        self.steps = 0

    def step(self, target_pos):
        m, d = self.m, self.d
        jac = np.zeros((6, m.nv))
        mujoco.mj_jacSite(m, d, jac[:3], jac[3:], self.site)
        jac = jac[:, self.dof]

        err = np.zeros(6)
        err[:3] = target_pos - d.site(self.site).xpos
        site_quat, site_quat_conj, err_quat = np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, d.site(self.site).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(err_quat, self.target_quat, site_quat_conj)
        mujoco.mju_quat2Vel(err[3:], err_quat, 1.0)

        dq = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING * np.eye(6), err)
        d.ctrl[:7] = np.clip(d.qpos[:7] + dq * INTEGRATION_DT, *m.actuator_ctrlrange[:7].T)
        mujoco.mj_step(m, d)

        self.steps += 1
        if self.writer is not None and self.steps % self.render_every == 0:
            self.writer.append_data(render(m, d, self.renderer))

    def move_to(self, goal, tol=0.006, max_steps=3000):
        start = self.d.site(self.site).xpos.copy()
        n = max(1, int(np.linalg.norm(goal - start) / MAX_STEP * 10))
        for i in range(max_steps):
            self.step(start + (goal - start) * min(1.0, (i + 1) / n))
            if i >= n and np.linalg.norm(goal - self.d.site(self.site).xpos) < tol:
                break

    def gripper(self, open_, steps=400):
        self.d.ctrl[7] = 255 if open_ else 0
        hold = self.d.site(self.site).xpos.copy()
        for _ in range(steps):
            self.step(hold)

    def pick_and_place(self, pick_xyz, place_xyz):
        grasp_z = max(pick_xyz[2] - 0.015, 0.012)   # VLM point lands on the top face; grasp below it
        above = 0.18
        self.gripper(True, 100)
        self.move_to(np.array([pick_xyz[0], pick_xyz[1], above]))
        self.move_to(np.array([pick_xyz[0], pick_xyz[1], grasp_z]))
        self.gripper(False)
        self.move_to(np.array([pick_xyz[0], pick_xyz[1], above]))
        self.move_to(np.array([place_xyz[0], place_xyz[1], above]))
        self.move_to(np.array([place_xyz[0], place_xyz[1], CUBE_HALF + 0.02]))
        self.gripper(True)
        self.move_to(np.array([place_xyz[0], place_xyz[1], above]))


# ============================================================
# Episode: perceive -> decide -> act -> verify
# ============================================================

def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:50]


def run_episode(mode, task, seed, vlm=None, decider=None, expected=None, out_root=OUT_DIR):
    """Run one episode and save its demo + result.json under out_root/<mode>/seed<N>_<task>/.
    `expected` = (pick, place) canonical names, used to score the decision."""
    perception, backend = MODES[mode]
    decider = decider or Decider(backend, vlm)
    out_dir = os.path.join(out_root, mode, f"seed{seed}_{slug(task)}")
    os.makedirs(out_dir, exist_ok=True)

    model, data = build_scene(np.random.default_rng(seed))
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    for _ in range(200):  # let cubes settle
        mujoco.mj_step(model, data)
    gt = perceive_gt(model, data)

    # 1) Perceive
    t0 = time.perf_counter()
    if perception == "qwen":
        scene = perceive_vlm(vlm, model, data, renderer, out_dir)
    else:
        scene = perceive_gt(model, data)
        Image.fromarray(render(model, data, renderer)).save(os.path.join(out_dir, "vlm_input.png"))
    t_perceive = time.perf_counter() - t0

    # Perception error: for each true cube, the closest estimate (cm)
    cube_err = {}
    for name in CUBES:
        dists = [np.linalg.norm(p[:2] - gt[name][:2]) for k, p in scene.items() if canonical(k, scene, gt) == name]
        cube_err[name] = round(min(dists) * 100, 2) if dists else None
    for k, p in scene.items():
        print(f"[Perceive:{perception}] {k:12s} -> {np.round(p, 3)}")

    # 2) Decide
    t0 = time.perf_counter()
    objects, targets = Decider.split(scene)
    decider.info = {}
    pick, place = decider.choose_pick_place(task, scene) if objects and targets else (None, None)
    decide_trace = dict(decider.info)
    t_decide = time.perf_counter() - t0
    pick_c, place_c = canonical(pick, scene, gt), canonical(place, scene, gt)
    print(f"[Decide:{decider.backend}] '{task}' -> pick='{pick}' ({pick_c}), place='{place}' ({place_c})")

    # 3) Act
    steps = 0
    if pick and place:
        with imageio.get_writer(os.path.join(out_dir, "pick_place.mp4"), fps=30) as writer:
            ctrl = Controller(model, data, renderer, writer)
            ctrl.pick_and_place(scene[pick], scene[place])
            steps = ctrl.steps

    # 4) Verify: physics ground truth + the backend's own judgement
    target_pick, target_place = expected or (pick_c, place_c)
    sim_success = False
    if target_pick in CUBES and target_place in MATS:
        cube = data.body(target_pick.replace(" ", "_")).xpos
        sim_success = bool(np.all(np.abs(cube[:2] - MATS[target_place][0]) < 0.07) and cube[2] < 0.05)

    final = render(model, data, renderer)
    Image.fromarray(final).save(os.path.join(out_dir, "final.png"))
    renderer.close()

    t0 = time.perf_counter()
    description = None
    if vlm is not None and perception == "qwen" and pick:
        description = vlm.ask(Image.fromarray(final), f"Where is the {pick} now? Is it on the {place}? "
                                                      "Answer in one or two sentences.", 128)
    decider.info = {}
    judged = decider.judge_success(task, final, description)
    judge_trace = dict(decider.info)
    t_verify = time.perf_counter() - t0
    print(f"[Verify] judged={judged} | simulator={sim_success}")

    result = {
        "mode": mode, "perception": perception, "decider": decider.backend, "task": task, "seed": seed,
        "pick": pick, "place": place, "pick_true": pick_c, "place_true": place_c,
        "expected_pick": expected[0] if expected else None, "expected_place": expected[1] if expected else None,
        "decision_correct": (pick_c, place_c) == tuple(expected) if expected else None,
        "sim_success": sim_success, "judged_success": judged,
        "judge_agrees": None if judged is None else judged == sim_success,
        "cube_err_cm": cube_err, "n_detected": len(scene),
        "t_perceive_s": round(t_perceive, 3), "t_decide_s": round(t_decide, 3), "t_verify_s": round(t_verify, 3),
        "sim_steps": steps, "vlm_description": description, "out_dir": os.path.relpath(out_dir, REPO_ROOT),
        "escalated_decide": decide_trace.get("escalated"), "escalated_judge": judge_trace.get("escalated"),
        "decide_trace": decide_trace, "judge_trace": judge_trace,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"[Saved] {out_dir}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="combined")
    ap.add_argument("--task", default="put the red cube on the purple mat")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    vlm = VLM() if MODES[args.mode][0] == "qwen" else None
    run_episode(args.mode, args.task, args.seed, vlm=vlm)


if __name__ == "__main__":
    main()
