"""
Two-robot collaboration in MuJoCo: a Franka Panda arm and a Unitree Go1 quadruped with a tray on its back,
coordinated by a VLM planner (GPT via the OpenAI Responses API, or Qwen3-VL-8B locally).

  - Franka: picks cubes from the floor, from the mats, or from the dog's tray (scripted differential IK, as in
    jmujoco_jev.py).
  - Go1: cannot grasp. It carries up to two cubes on its tray and walks between named spots (dock, orange zone,
    blue zone, charging pad). Walking uses MuJoCo Playground's pretrained Go1 joystick policy (ONNX, sim2sim). A
    waypoint follower turns "go to X" into velocity commands (vx, vy, yaw rate) for that policy.
  - VLM: each step it sees one image (overview of the whole arena + side view of the arm) and the log of attempted
    actions. It returns the remaining plan as typed actions, and only the first one runs (closed loop, as in
    jmujoco_api_llm.py). Grasp targets come from simulator poses, so the VLM only plans.

The arm can load or unload the tray only while the dog is at the dock. Otherwise the action is rejected, and the
rejection is reported back in the action log.

Setup (once, from repo root):
    git -C external/mujoco_menagerie sparse-checkout add unitree_go1
    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_playground external/mujoco_playground
    git -C external/mujoco_playground sparse-checkout set mujoco_playground/_src/locomotion/go1 mujoco_playground/experimental/sim2sim

Usage (from repo root, isaac_venv; needs onnxruntime):
    # scripted oracle plan (no LLM), to check the robots and the scene
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_go1_collab.py run --model script --task 0
    # VLM planner
    export OPENAI_API_KEY=...
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_go1_collab.py run --model gpt-5.5 --task 2 --seed 0
    MUJOCO_GL=egl isaac_venv/bin/python rob_envs/MuJoCo/jmujoco_go1_collab.py sweep --models gpt-5.4-mini qwen --seeds 0 1
Outputs: outputs/mujoco_go1/<model>/seed<N>_task<K>/{episode.mp4, step_<k>.png, final.png, result.json}, results.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jmujoco_api_llm as L  # noqa: E402  (planners, cameras, mats; also sets J.IMG_W/H to 960x540)
import jmujoco_jev as J  # noqa: E402  (IK controller, JSON parsing)

OUT_DIR = os.path.join(J.REPO_ROOT, "outputs/mujoco_go1")
PLAYGROUND = os.path.join(J.REPO_ROOT, "external/mujoco_playground/mujoco_playground")
GO1_XML = os.path.join(PLAYGROUND, "_src/locomotion/go1/xmls/go1_mjx_feetonly.xml")
GO1_MESHES = os.path.join(J.REPO_ROOT, "external/mujoco_menagerie/unitree_go1/assets")
GO1_POLICY = os.path.join(PLAYGROUND, "experimental/sim2sim/onnx/go1_policy.onnx")

# Go1 policy settings, as in Playground's play_go1_joystick.py
GO1_HOME_Q = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)
GO1_HOME_Z = 0.278
POLICY_DT, ACTION_SCALE = 0.02, 0.5
MAX_VX, MAX_VY, MAX_WZ = 0.6, 0.4, 1.0  # stay well inside the training command range (1.5, 0.8, 2pi)
MIN_SPEED = 0.35

CUBES = {k: L.CUBES[k] for k in ("red cube", "yellow cube", "green cube", "blue cube")}
MATS = L.MATS
# Where the dog can walk to (trunk xy; the dog always faces +x). All spots have y <= -0.6, so straight paths
# between them stay clear of the arm, the mats and the floor cubes.
SPOTS = {"dock": (0.50, -0.60), "orange zone": (1.70, -0.60), "blue zone": (1.15, -1.35),
         "charging pad": (1.80, -1.35)}
SPOT_RGBA = {"dock": [0.75, 0.75, 0.75, 1], "orange zone": [1.0, 0.55, 0.1, 1], "blue zone": [0.2, 0.45, 0.95, 1],
             "charging pad": [0.35, 0.35, 0.35, 1]}
ZONE_HALF = 0.32
DOCK_TOL = 0.08
# Tray on the dog's back: two slots along the dog's x axis (= world x at the dock, the axis the fingers open on)
TRAY_Z, TRAY_IN, TRAY_WALL = 0.065, (0.12, 0.065), 0.012
TRAY_SLOTS = [(-0.045, 0.0), (0.045, 0.0)]
ARM_REACH = 0.82
MAX_STEPS = 12

# (instruction, expected final location per cube (None = must not end on a mat or in a zone),
#  dog start (spot, cubes on its tray), oracle plan for --model script)
TASKS = [
    ("deliver the red cube to the orange zone",
     {"red cube": "orange zone"}, ("charging pad", []),
     [("dog", "dock"), ("arm", "red cube", "dog tray"), ("dog", "orange zone")]),
    ("the dog is carrying the blue cube; put it on the purple mat",
     {"blue cube": "purple mat"}, ("blue zone", ["blue cube"]),
     [("dog", "dock"), ("arm", "blue cube", "purple mat")]),
    ("deliver the yellow and green cubes to the blue zone",
     {"yellow cube": "blue zone", "green cube": "blue zone"}, ("charging pad", []),
     [("dog", "dock"), ("arm", "yellow cube", "dog tray"), ("arm", "green cube", "dog tray"), ("dog", "blue zone")]),
    ("put the cube the dog is carrying on the white mat, then deliver the red cube to the orange zone",
     {"blue cube": "white mat", "red cube": "orange zone"}, ("orange zone", ["blue cube"]),
     [("dog", "dock"), ("arm", "blue cube", "white mat"), ("arm", "red cube", "dog tray"), ("dog", "orange zone")]),
    ("send the cube that is the color of grass to the zone that is the color of the sky",
     {"green cube": "blue zone"}, ("charging pad", []),
     [("dog", "dock"), ("arm", "green cube", "dog tray"), ("dog", "blue zone")]),
]


# ============================================================
# Scene: Panda scene + Go1 (attached with prefix "go1/") + tray, mats, zones, cubes
# ============================================================

def load_go1_spec():
    """Playground's feet-only Go1, with mesh paths pointed at the menagerie checkout."""
    if not (os.path.exists(GO1_XML) and os.path.exists(GO1_POLICY) and os.path.isdir(GO1_MESHES)):
        sys.exit("Go1 model/policy not found; run the setup commands in this file's docstring.")
    xml = open(GO1_XML).read()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{GO1_MESHES}"', xml)
    xml = re.sub(r'file="[^"]*/([^/"]+\.stl)"', r'file="\1"', xml)
    spec = mujoco.MjSpec.from_string(xml)
    spec.delete(spec.light("spotlight"))
    # The Panda scene's physics options win on attach; match them here so attaching doesn't warn
    # (the policy walks fine at 2 ms implicitfast instead of its training 4 ms Euler)
    spec.option.timestep, spec.option.iterations, spec.option.ls_iterations = 0.002, 100, 50
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    trunk = spec.body("trunk")
    trunk.pos = [0, 0, 0]
    # Tray: a thin plate with low walls, rigidly on the trunk. Collides with cubes and fingers only.
    (hx, hy), t = TRAY_IN, 0.004
    tray = dict(type=mujoco.mjtGeom.mjGEOM_BOX, rgba=[0.55, 0.4, 0.25, 1], contype=1, conaffinity=1, condim=4,
                friction=[1.5, 0.01, 0.001], mass=0.03, group=0)
    trunk.add_geom(name="tray", pos=[0, 0, TRAY_Z], size=[hx + t, hy + t, t], **tray)
    for sx in (-1, 1):
        trunk.add_geom(pos=[sx * (hx + t), 0, TRAY_Z + TRAY_WALL / 2], size=[t, hy + t, TRAY_WALL / 2], **tray)
    for sy in (-1, 1):
        trunk.add_geom(pos=[0, sy * (hy + t), TRAY_Z + TRAY_WALL / 2], size=[hx + t, t, TRAY_WALL / 2], **tray)
    return spec


def add_cube(spec, name, pos):
    body = spec.worldbody.add_body(name=name.replace(" ", "_"), pos=pos)
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[J.CUBE_HALF] * 3, rgba=CUBES[name],
                  mass=0.05, friction=[1.5, 0.01, 0.001], condim=4)


def build_scene(rng, task_id):
    _, _, (dog_spot, carried), _ = TASKS[task_id]
    spec = mujoco.MjSpec.from_file(J.PANDA_XML)
    spec.delete(spec.key("home"))
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005] * 3, rgba=[1, 0, 0, 0])
    for body in spec.bodies:
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1.0
    for name, (xy, rgba) in MATS.items():
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*xy, 0.001],
                                size=[L.MAT_HALF, L.MAT_HALF, 0.001], rgba=rgba, contype=0, conaffinity=0)
    for name, xy in SPOTS.items():
        size = [0.12, 0.12] if name == "dock" else [ZONE_HALF, ZONE_HALF]
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*xy, 0.0005], size=[*size, 0.0005],
                                rgba=SPOT_RGBA[name], contype=0, conaffinity=0)

    # Go1 at its start spot (Panda joints/actuators stay first in qpos/ctrl, which J.Controller assumes)
    dog_xy = np.array(SPOTS[dog_spot])
    frame = spec.worldbody.add_frame(pos=[*dog_xy, GO1_HOME_Z])
    frame.attach_body(load_go1_spec().body("trunk"), "go1/", "")

    floor_cubes = [c for c in CUBES if c not in carried]
    for name, xy in zip(floor_cubes, rng.permutation(L.CUBE_SLOTS)):
        add_cube(spec, name, [*(xy + rng.uniform(-0.015, 0.015, 2)), J.CUBE_HALF])
    for name, slot in zip(carried, TRAY_SLOTS):
        add_cube(spec, name, [dog_xy[0] + slot[0], dog_xy[1] + slot[1], GO1_HOME_Z + TRAY_Z + 0.004 + J.CUBE_HALF])

    # Overview: high, from behind the arm, so the zones are at the top of the image and the arm at the bottom.
    # Side view of the arm: experiment 2's camera mirrored to the +y side, where the docked dog does not block it.
    lookat = np.array([0.95, -0.6, 0.0])
    cam_pos = lookat + np.array([-1.45, 0.0, 2.1])
    spec.worldbody.add_camera(name="overview", pos=cam_pos, quat=J.lookat_quat(cam_pos, lookat), fovy=55)
    lookat, az, el, dist = np.array([0.30, 0.0, 0.30]), np.deg2rad(-120), np.deg2rad(-20), 1.75
    cam_pos = lookat - dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    spec.worldbody.add_camera(name="front", pos=cam_pos, quat=J.lookat_quat(cam_pos, lookat), fovy=45)

    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = J.IMG_W, J.IMG_H
    model.vis.headlight.diffuse[:], model.vis.headlight.ambient[:] = 0.3, 0.25
    model.light_diffuse[:], model.light_specular[:], model.mat_specular[:] = 0.5, 0.1, 0.1
    data = mujoco.MjData(model)
    data.qpos[:7], data.qpos[7:9] = J.HOME_Q, 0.04
    data.ctrl[:7], data.ctrl[7] = J.HOME_Q, 255
    dog = Go1(model)
    data.qpos[dog.qadr] = GO1_HOME_Q
    data.ctrl[dog.act] = GO1_HOME_Q
    mujoco.mj_forward(model, data)
    return model, data, dog


# ============================================================
# Go1: pretrained joystick policy + waypoint follower
# ============================================================

class Go1:
    """Runs Playground's Go1 joystick policy inside mj_step (via mjcb_control) at 50 Hz, tracking self.command."""

    def __init__(self, model):
        import onnxruntime as rt
        self.policy = rt.InferenceSession(GO1_POLICY, providers=["CPUExecutionProvider"])
        legs = [f"go1/{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
        self.qadr = np.array([model.joint(j).qposadr[0] for j in legs])
        self.vadr = np.array([model.joint(j).dofadr[0] for j in legs])
        self.act = np.array([model.actuator(j.replace("_joint", "")).id for j in legs])
        self.trunk = model.body("go1/trunk").id
        self.imu = model.site("go1/imu").id
        self.command = np.zeros(3)  # vx, vy (body frame), yaw rate
        self.last_action = np.zeros(12, np.float32)
        self.next_t = 0.0

    def control(self, model, data):
        if data.time + 1e-9 < self.next_t:
            return
        self.next_t = data.time + POLICY_DT
        gravity = data.site_xmat[self.imu].reshape(3, 3).T @ np.array([0, 0, -1])
        obs = np.hstack([data.sensor("go1/local_linvel").data, data.sensor("go1/gyro").data, gravity,
                         data.qpos[self.qadr] - GO1_HOME_Q, data.qvel[self.vadr], self.last_action, self.command])
        self.last_action = self.policy.run(["continuous_actions"], {"obs": obs.astype(np.float32)[None]})[0][0]
        data.ctrl[self.act] = self.last_action * ACTION_SCALE + GO1_HOME_Q

    def pose(self, data):
        xmat = data.xmat[self.trunk].reshape(3, 3)
        return data.xpos[self.trunk].copy(), np.arctan2(xmat[1, 0], xmat[0, 0])

    def tray_point(self, data, slot=(0.0, 0.0)):
        """World position of a tray slot's surface."""
        return data.xpos[self.trunk] + data.xmat[self.trunk].reshape(3, 3) @ np.array([*slot, TRAY_Z + 0.004])

    def walk_to(self, sim, xy, tol=0.04, timeout=25.0):
        """Holonomic P-control on the trunk position, heading held at 0. Returns True if it arrived."""
        data, dt = sim.d, sim.m.opt.timestep
        t0, arrived = data.time, False
        while data.time - t0 < timeout:
            pos, yaw = self.pose(data)
            err = np.asarray(xy) - pos[:2]
            if np.linalg.norm(err) < tol and abs(yaw) < 0.05:
                arrived = True
                break
            c, s = np.cos(yaw), np.sin(yaw)
            e_body = np.array([c * err[0] + s * err[1], -s * err[0] + c * err[1]])
            # The policy barely tracks very small commands, so keep at least MIN_SPEED until arrival
            dist = np.linalg.norm(e_body)
            v = e_body / max(dist, 1e-6) * np.clip(1.5 * dist, MIN_SPEED, MAX_VX) if dist > tol else np.zeros(2)
            self.command[:] = [np.clip(v[0], -MAX_VX, MAX_VX), np.clip(v[1], -MAX_VY, MAX_VY),
                               np.clip(-2.0 * yaw, -MAX_WZ, MAX_WZ)]
            sim.step(round(POLICY_DT / dt))
        self.command[:] = 0
        sim.step(round(1.0 / dt))  # settle into a stand before anyone reaches over it
        return arrived

    def spot(self, data):
        pos, _ = self.pose(data)
        name = min(SPOTS, key=lambda s: np.linalg.norm(pos[:2] - SPOTS[s]))
        return name if np.linalg.norm(pos[:2] - SPOTS[name]) < (DOCK_TOL if name == "dock" else ZONE_HALF) else None


# ============================================================
# Recording + arm control (every mj_step goes through Sim.step so the video sees both robots)
# ============================================================

class Sim:
    def __init__(self, model, data, renderer, writer, cams=("overview", "front")):
        self.m, self.d, self.renderer, self.writer, self.cams = model, data, renderer, writer, cams
        self.every = max(1, int(1 / 30 / model.opt.timestep))
        self.n = 0

    def frame(self):
        return np.concatenate([L.render_cam(self.m, self.d, self.renderer, c) for c in self.cams], axis=1)

    def tick(self):
        self.n += 1
        if self.writer is not None and self.n % self.every == 0:
            self.writer.append_data(self.frame())

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)
            self.tick()


class Arm(J.Controller):
    def __init__(self, sim):
        super().__init__(sim.m, sim.d, sim.renderer, None)
        self.sim = sim

    def step(self, target_pos):
        super().step(target_pos)  # J.Controller.step does the mj_step
        self.sim.tick()

    def transfer(self, pick_xyz, place_xyz):
        """Pick a cube whose center is at pick_xyz; set it down on the surface at place_xyz (z = surface height)."""
        above = max(pick_xyz[2], place_xyz[2]) + 0.18
        grasp_z = max(pick_xyz[2] - 0.005, 0.012)
        self.gripper(True, 100)
        self.move_to(np.array([pick_xyz[0], pick_xyz[1], above]))
        self.move_to(np.array([pick_xyz[0], pick_xyz[1], grasp_z]))
        self.gripper(False)
        self.move_to(np.array([pick_xyz[0], pick_xyz[1], above]))
        self.move_to(np.array([place_xyz[0], place_xyz[1], above]))
        self.move_to(np.array([place_xyz[0], place_xyz[1], place_xyz[2] + J.CUBE_HALF + 0.015]))
        self.gripper(True)
        self.move_to(np.array([place_xyz[0], place_xyz[1], above]))


# ============================================================
# State, prompt, planners
# ============================================================

def cube_pos(data, name):
    return data.body(name.replace(" ", "_")).xpos.copy()


def location(data, name):
    """'purple mat' / 'white mat' / a zone / None, from the cube's simulator pose."""
    xyz = cube_pos(data, name)
    if mat := L.mat_of(xyz):
        return mat
    for zone, xy in SPOTS.items():
        if zone != "dock" and np.all(np.abs(xyz[:2] - xy) < ZONE_HALF):
            return zone
    return None


def on_tray(data, dog, name):
    xyz, tray = cube_pos(data, name), dog.tray_point(data)
    return bool(np.all(np.abs(xyz[:2] - tray[:2]) < np.array(TRAY_IN) + 0.01) and abs(xyz[2] - tray[2]) < 0.06)


def build_prompt(task, history):
    log = "\n".join(f"{i + 1}. {fmt_action(a)} -> {a['outcome']}" for i, a in enumerate(history)) or "none yet"
    return (
        "You coordinate two robots that work together in a simulated room.\n"
        "- ARM: a fixed robot arm (left of the scene). It picks one small cube at a time and places it on the "
        "\"purple mat\", the \"white mat\" or the \"dog tray\". It reaches cubes on the floor near it, on the mats, "
        "and on the dog's tray ONLY while the dog stands at the dock (the gray square next to the arm).\n"
        "- DOG: a four-legged robot with a brown tray on its back that holds at most two cubes. It cannot grasp; it "
        "only walks to one of: \"dock\", \"orange zone\", \"blue zone\", \"charging pad\" (dark gray square). Cubes on "
        "its tray travel with it. A cube is delivered to a zone when the dog stands there carrying it.\n"
        "Only the arm moves cubes on or off the tray: to take a cube off the dog, first send the dog to the dock, "
        "then have the arm pick that cube and place it where it belongs.\n"
        "You get ONE image with two views side by side: LEFT, a high overview of the whole room seen from behind the "
        "arm (the arm and the mats at the bottom left, the zones and the charging pad at the top); RIGHT, a closer "
        "side view of the arm, the mats, the floor cubes and the dock.\n"
        f"Cubes in the scene: {', '.join(CUBES)}.\n"
        f"Instruction: {task}\n"
        f"Actions attempted so far (with what happened):\n{log}\n\n"
        "Judge from the image where every cube and the dog are now, then give the remaining plan to satisfy the "
        "instruction. Never move a cube the instruction does not mention. Answer ONLY with JSON:\n"
        '{"remaining_plan": [{"robot": "arm", "pick": "<color> cube", "place": "purple mat" or "white mat" or '
        '"dog tray"} or {"robot": "dog", "go_to": "dock" or "orange zone" or "blue zone" or "charging pad"}, ...], '
        '"done": true if the instruction is already satisfied else false}'
    )


def fmt_action(a):
    return f"dog go_to {a.get('go_to')}" if a.get("robot") == "dog" else f"arm pick {a.get('pick')}, place on {a.get('place')}"


class ScriptedPlanner:
    """Oracle plan from TASKS, one action per call; checks the scene/robots without any LLM."""
    name = "script"

    def __init__(self):
        self.plan = []

    def reset(self, task_id):
        self.plan = [{"robot": "dog", "go_to": a[1]} if a[0] == "dog" else {"robot": "arm", "pick": a[1], "place": a[2]}
                     for a in TASKS[task_id][3]]

    def __call__(self, img, prompt, memory=()):
        out = {"remaining_plan": self.plan, "done": not self.plan}
        self.plan = self.plan[1:]
        return json.dumps(out), {"in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0}


def make_planner(name, effort=None, vlm=None):
    return ScriptedPlanner() if name == "script" else L.make_planner(name, effort, vlm)


# ============================================================
# Episode: observe -> plan -> execute first action -> repeat
# ============================================================

def execute(action, data, dog, arm, sim, home_tcp):
    """Run one typed action. Returns its outcome text (shown to the planner in the next step's action log)."""
    if action.get("robot") == "dog":
        spot = L.snap(action.get("go_to"), list(SPOTS))
        if spot is None:
            return f"rejected: the dog can only go to {', '.join(SPOTS)}, not {action.get('go_to')!r}"
        arm.move_to(home_tcp)  # keep the arm clear of the dog
        ok = dog.walk_to(sim, SPOTS[spot])
        return "done" if ok else "the dog did not reach it in time"

    pick, place = L.snap(action.get("pick"), list(CUBES)), L.snap(action.get("place"), list(MATS) + ["dog tray"])
    if pick is None or place is None:
        return f"rejected: unknown cube or place ({action.get('pick')!r}, {action.get('place')!r})"
    at_dock = dog.spot(data) == "dock"
    if on_tray(data, dog, pick) and not at_dock:
        return "rejected: that cube is on the dog's tray and the dog is not at the dock"
    if place == "dog tray" and not at_dock:
        return "rejected: the dog is not at the dock"
    pick_xyz = cube_pos(data, pick)
    if np.linalg.norm(pick_xyz[:2]) > ARM_REACH:
        return "rejected: that cube is out of the arm's reach"
    if place == "dog tray":
        taken = [cube_pos(data, c) for c in CUBES if c != pick and on_tray(data, dog, c)]
        free = [s for s in TRAY_SLOTS if all(np.linalg.norm(dog.tray_point(data, s)[:2] - p[:2]) > 0.04 for p in taken)]
        if not free:
            return "rejected: the dog's tray is full"
        target = dog.tray_point(data, free[0])
    else:
        fill = sum(location(data, c) == place for c in CUBES if c != pick)
        slot = L.MAT_SLOTS[fill % len(L.MAT_SLOTS)]
        target = np.array([MATS[place][0][0] + slot[0], MATS[place][0][1] + slot[1], 0.0])
    arm.transfer(pick_xyz, target)
    arm.move_to(home_tcp)
    landed = "dog tray" if on_tray(data, dog, pick) else location(data, pick)
    return "done" if landed == place else "executed, but the cube did not end up there"


def run_episode(planner, task_id, seed, out_root=OUT_DIR):
    task, expected, _, _ = TASKS[task_id]
    out_dir = os.path.join(out_root, planner.name, f"seed{seed}_task{task_id}")
    os.makedirs(out_dir, exist_ok=True)
    if isinstance(planner, ScriptedPlanner):
        planner.reset(task_id)

    model, data, dog = build_scene(np.random.default_rng(seed), task_id)
    renderer = mujoco.Renderer(model, height=J.IMG_H, width=J.IMG_W)
    writer = imageio.get_writer(os.path.join(out_dir, "episode.mp4"), fps=30)
    sim = Sim(model, data, renderer, writer)
    arm = Arm(sim)
    mujoco.set_mjcb_control(dog.control)
    history, steps_log = [], []
    t_episode = time.perf_counter()
    try:
        sim.step(int(1.0 / model.opt.timestep))  # let cubes settle and the dog find its stance
        home_tcp = data.site(arm.site).xpos.copy()
        for step in range(MAX_STEPS):
            img = sim.frame()
            Image.fromarray(img).save(os.path.join(out_dir, f"step_{step}.png"))
            t0 = time.perf_counter()
            try:
                raw, usage = planner(img, build_prompt(task, history))
            except Exception as e:  # API errors end the episode but are logged
                steps_log.append({"step": step, "error": f"{type(e).__name__}: {e}"})
                print(f"[step {step}] planner error: {e}")
                break
            latency = time.perf_counter() - t0
            out = J.parse_json(raw, "{}") or {}
            plan = out.get("remaining_plan") or []
            log = {"step": step, "latency_s": round(latency, 2), **usage, "done": out.get("done"), "plan": plan,
                   "raw": raw if not out else None}
            steps_log.append(log)
            if out.get("done") or not plan:
                print(f"[step {step}] planner says done ({latency:.1f}s)")
                break
            action = plan[0] if isinstance(plan[0], dict) else {}
            outcome = execute(action, data, dog, arm, sim, home_tcp)
            history.append({**action, "outcome": outcome})
            log.update(action=action, outcome=outcome, dog_spot=dog.spot(data),
                       cubes={c: ("dog tray" if on_tray(data, dog, c) else location(data, c)) for c in CUBES})
            print(f"[step {step}] {fmt_action(action)} -> {outcome} | dog at {dog.spot(data)} | "
                  f"plan {len(plan)} left | {latency:.1f}s")
        final = sim.frame()
    finally:
        mujoco.set_mjcb_control(None)
        writer.close()
    Image.fromarray(final).save(os.path.join(out_dir, "final.png"))
    renderer.close()

    final_loc = {c: location(data, c) for c in CUBES}
    correct = {c: final_loc[c] == expected.get(c) for c in CUBES}
    result = {
        "model": planner.name, "task_id": task_id, "task": task, "seed": seed,
        "success": all(correct.values()), "cube_accuracy": round(float(np.mean(list(correct.values()))), 3),
        "final_locations": final_loc, "expected": {c: expected.get(c) for c in CUBES},
        "dog_final_spot": dog.spot(data), "dog_fell": bool(dog.pose(data)[0][2] < 0.15),
        "n_actions": len(history), "n_planner_calls": len(steps_log),
        "rejected_actions": sum(a["outcome"].startswith("rejected") for a in history),
        "planner_said_done": bool(steps_log and steps_log[-1].get("done")),
        "errors": [s["error"] for s in steps_log if "error" in s],
        "planner_latency_s": [s["latency_s"] for s in steps_log if "latency_s" in s],
        "cost_usd": round(sum(s.get("cost_usd", 0) for s in steps_log), 5),
        "episode_s": round(time.perf_counter() - t_episode, 1), "sim_s": round(data.time, 1), "steps": steps_log,
        "out_dir": os.path.relpath(out_dir, J.REPO_ROOT),
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[result] success={result['success']} cube_acc={result['cube_accuracy']} final={final_loc} "
          f"actions={result['n_actions']} cost=${result['cost_usd']:.4f} -> {out_dir}")
    return result


def write_csv(rows):
    path = os.path.join(OUT_DIR, "results.csv")
    keys = ["model", "task_id", "seed", "success", "cube_accuracy", "n_actions", "n_planner_calls",
            "rejected_actions", "dog_fell", "cost_usd", "episode_s", "out_dir"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one episode")
    r.add_argument("--model", default="script", help="script | gpt-5.5 | gpt-5.4 | gpt-5.4-mini | gpt-5.4-nano | qwen")
    r.add_argument("--task", type=int, default=0, help=f"task index 0-{len(TASKS) - 1}")
    r.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sweep", help="models x tasks x seeds, then results.csv")
    s.add_argument("--models", nargs="+", default=["script", "gpt-5.4-mini", "gpt-5.5", "qwen"])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    s.add_argument("--tasks", nargs="+", type=int, default=list(range(len(TASKS))))
    for p in (r, s):
        p.add_argument("--effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                       help="GPT reasoning effort (default: model default)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    models = [args.model] if args.cmd == "run" else args.models
    if any(m not in ("qwen", "script") for m in models) and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. Run `export OPENAI_API_KEY=...` first.")
    if args.cmd == "run":
        run_episode(make_planner(args.model, args.effort), args.task, args.seed)
        return
    vlm = J.VLM() if "qwen" in models else None
    for m in models:
        planner = make_planner(m, args.effort, vlm)
        for seed in args.seeds:
            for t in args.tasks:
                print(f"\n===== {m} | seed {seed} | task {t}: {TASKS[t][0]}")
                try:
                    run_episode(planner, t, seed)
                except Exception as e:
                    print(f"[ERROR] {type(e).__name__}: {e}")
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_DIR, "*", "*", "result.json")))]
    print(f"\nCSV -> {write_csv(rows)}")


if __name__ == "__main__":
    main()
