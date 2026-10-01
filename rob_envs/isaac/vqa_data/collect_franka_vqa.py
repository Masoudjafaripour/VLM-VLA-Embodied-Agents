"""Collect a vision-language-action (VQA-style) dataset with a scripted Franka expert in Isaac Lab.

Scene : Franka Panda + 3 cubes (red / green / blue) at random positions, one fixed RGB camera.
Task  : "pick up the <color> cube". A scripted IK expert does approach -> descend -> grasp -> lift.
Saved : only successful episodes.

Output (default outputs/isaac/vqa_franka/):
    images/ep0000_t000.png ...    224x224 RGB frames (20 Hz)
    actions.jsonl                 one line per step  -> (image, instruction, action)        [policy / VLA data]
    qa.jsonl                      perception Q&A on the first frame of each episode         [VQA data]
    meta.json                     action format, binning, camera, counts

Action (7-D, robot base frame = world frame here):
    [dx, dy, dz, droll, dpitch, dyaw, gripper]   dx..dz in m per step, rotation deltas are 0 (top-down grasp),
    gripper +1 = open, -1 = close.  `action_tokens` = each dim binned to 0..255 (RT-2 / OpenVLA style).

Run:
    python rob_envs/isaac/vqa_data/collect_franka_vqa.py --episodes 50
    python rob_envs/isaac/vqa_data/collect_franka_vqa.py --episodes 5 --viz kit    # watch it
    python rob_envs/isaac/vqa_data/collect_franka_vqa.py --episodes 2 --debug      # phase logs + GIF per attempt
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=20, help="number of successful episodes to save")
parser.add_argument("--out", default="outputs/isaac/vqa_franka")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_tries", type=int, default=0, help="stop after this many attempts (0 = 3x episodes)")
parser.add_argument("--debug", action="store_true", help="print expert phase diagnostics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app  # must launch before other isaaclab imports

import imageio.v3 as iio
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# ----------------------------------------------------------------------------- config
SIM_DT, DECIMATION = 0.01, 5             # physics 100 Hz, control + recording 20 Hz
CUBE_SIZE = 0.045
COLORS = {"red": (0.85, 0.1, 0.1), "green": (0.1, 0.7, 0.1), "blue": (0.1, 0.2, 0.85)}
WORKSPACE = ((0.35, 0.65), (-0.25, 0.25))  # x, y range for cube centers [m]
MIN_CUBE_DIST = 0.12
HAND_TO_TIP = 0.1034                     # panda_hand origin -> fingertip center (TCP) [m]
MAX_STEP = 0.01                          # max EE translation per control step [m] (0.2 m/s)
POS_BIN_RANGE = 0.02                     # dx/dy/dz binned over [-0.02, 0.02] m
CAM_EYE, CAM_LOOKAT = (1.25, 0.0, 0.65), (0.45, 0.0, 0.05)
INSTRUCTIONS = ["pick up the {c} cube", "grasp the {c} cube and lift it", "lift the {c} block"]

device = args.device or "cuda:0"
rng = np.random.default_rng(args.seed)

# ----------------------------------------------------------------------------- scene
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device=device))
sim.set_camera_view(CAM_EYE, CAM_LOOKAT)
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)
cfg = sim_utils.DomeLightCfg(intensity=2500.0)
cfg.func("/World/light", cfg)

robot = Articulation(FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Franka"))
cubes = {
    name: RigidObject(
        RigidObjectCfg(
            prim_path=f"/World/Cube_{name}",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.3 * i - 0.3, CUBE_SIZE / 2)),
        )
    )
    for i, (name, rgb) in enumerate(COLORS.items())
}
camera = Camera(
    CameraCfg(
        prim_path="/World/Camera",
        width=args.img_size,
        height=args.img_size,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
    )
)

sim.reset()
camera.set_world_poses_from_view(torch.tensor([CAM_EYE], device=device), torch.tensor([CAM_LOOKAT], device=device))

ARM = list(range(7))
HAND = robot.find_bodies("panda_hand")[0][0]
JACOBI_HAND = HAND - 1  # fixed-base robot: Jacobian skips the root body
ik = DifferentialIKController(
    DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), 1, device
)
default_q = robot.data.default_joint_pos.torch.clone()


# ----------------------------------------------------------------------------- helpers
def ee_pose():
    return robot.data.body_pos_w.torch[:, HAND].clone(), robot.data.body_quat_w.torch[:, HAND].clone()


def step_sim(arm_q, gripper_open, n=DECIMATION):
    """Hold arm joint targets + gripper command for n physics steps."""
    target = robot.data.joint_pos.torch.clone()
    target[:, ARM] = arm_q
    target[:, 7:] = 0.04 if gripper_open else 0.0
    for _ in range(n):
        robot.actuators.target_command.set_position_index(value=target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        for c in cubes.values():
            c.update(SIM_DT)
    camera.update(SIM_DT * n)


def sample_cube_xy():
    while True:
        xy = np.stack([rng.uniform(*WORKSPACE[0], 3), rng.uniform(*WORKSPACE[1], 3)], 1)
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(3)
        if d.min() > MIN_CUBE_DIST:
            return xy


def ik_step(pos, quat, ee_p, ee_q, gripper_open):
    """One control step: IK toward EE pose (pos, quat), then hold for DECIMATION physics steps."""
    ik.set_command(torch.cat([pos.view(1, 3), quat.view(1, 4)], -1))
    jac = robot.data.body_link_jacobian_w.torch[:, JACOBI_HAND, :, :][:, :, ARM]
    step_sim(ik.compute(ee_p, ee_q, jac, robot.data.joint_pos.torch[:, ARM]), gripper_open)


def reset_episode():
    """Robot to home pose, cubes to random non-overlapping poses. Returns {color: yaw}."""
    robot.write_joint_position_to_sim_index(position=default_q)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(default_q))
    yaws = {}
    for (name, c), (x, y) in zip(cubes.items(), sample_cube_xy()):
        yaw = yaws[name] = rng.uniform(-np.pi / 4, np.pi / 4)
        pose = [[x, y, CUBE_SIZE / 2, 0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]]  # quat is xyzw
        pose = torch.tensor(pose, dtype=torch.float32, device=device)
        c.write_root_pose_to_sim_index(root_pose=pose)
        c.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=device))
    ik.reset()
    for _ in range(4):  # let things settle and the renderer catch up
        step_sim(default_q[:, ARM], True)
    return yaws


def to_tokens(action):
    """Bin a 7-D action to 0..255 ints (pos over +-POS_BIN_RANGE, rot over +-pi, gripper over +-1)."""
    lo = np.array([-POS_BIN_RANGE] * 3 + [-np.pi] * 3 + [-1.0])
    hi = -lo
    return np.clip(np.round((action - lo) / (hi - lo) * 255), 0, 255).astype(int).tolist()


def project(p_w):
    """World point -> pixel (u, v) in the camera image."""
    K = camera.data.intrinsic_matrices.torch[0].cpu().numpy()
    R = matrix_from_quat(camera.data.quat_w_ros.torch)[0].cpu().numpy()  # cam(ROS) -> world
    p_c = R.T @ (np.asarray(p_w) - camera.data.pos_w.torch[0].cpu().numpy())
    u, v = (K @ p_c)[:2] / p_c[2]
    return int(round(u)), int(round(v))


def perception_qa(image_path, ee_p):
    """Ground-truth questions about the first frame."""
    pos = {n: c.data.root_pos_w.torch[0].cpu().numpy() for n, c in cubes.items()}
    qa = []
    for n, p in pos.items():
        u, v = project(p)
        qa.append((f"Where is the {n} cube in the image? Answer in pixel coordinates (x, y).", f"({u}, {v})"))
        qa.append((f"What is the 3D position of the {n} cube in meters?", f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"))
    closest = min(pos, key=lambda n: np.linalg.norm(pos[n] - ee_p))
    leftmost = max(pos, key=lambda n: pos[n][1])  # +y is image-left from this camera
    qa.append(("Which cube is closest to the gripper?", closest))
    qa.append(("Which cube is on the left side of the image?", leftmost))
    qa.append(("How many cubes are on the table?", str(len(pos))))
    qa.append(("Is the gripper holding an object?", "no"))
    return [{"image": image_path, "question": q, "answer": a} for q, a in qa]


# ----------------------------------------------------------------------------- expert episode
def run_episode(color, first_image):
    """Scripted pick. Returns (steps, qa, success); each step is (rgb, action[7])."""
    yaws = reset_episode()
    # top-down gripper, rotated to the cube's yaw so the fingers close on its faces (not recorded)
    yaw = torch.tensor([yaws[color]], device=device)
    hand_quat = quat_from_euler_xyz(torch.full_like(yaw, np.pi), torch.zeros_like(yaw), yaw)
    start = ee_pose()[0][0]
    for _ in range(40):
        ik_step(start, hand_quat, *ee_pose(), True)
    ee_p = ee_pose()[0]
    qa = perception_qa(first_image, ee_p[0].cpu().numpy())
    cube = cubes[color]
    c = cube.data.root_pos_w.torch[0].clone()
    above = c + torch.tensor([0, 0, HAND_TO_TIP + 0.12], device=device)
    grasp = c + torch.tensor([0, 0, HAND_TO_TIP - 0.005], device=device)
    lift = c + torch.tensor([0, 0, HAND_TO_TIP + 0.20], device=device)
    # (target, gripper open?, max control steps, stop early once reached?)
    phases = [(above, True, 60, True), (grasp, True, 30, True), (grasp, False, 10, False), (lift, False, 40, True)]

    steps = []
    setpoint = ee_p[0].clone()  # commanded EE position; the action is how much it moves each step
    for goal, open_, max_steps, stop_at_goal in phases:
        for _ in range(max_steps):
            ee_p, ee_q = ee_pose()
            if stop_at_goal and torch.linalg.norm(goal - ee_p[0]) < 0.008:
                break
            delta = goal - setpoint
            move = delta * min(1.0, MAX_STEP / max(torch.linalg.norm(delta).item(), 1e-6))
            setpoint = setpoint + move
            action = np.zeros(7, dtype=np.float32)
            action[:3] = move.cpu().numpy()
            action[6] = 1.0 if open_ else -1.0
            rgb = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy()
            steps.append((rgb, action))
            ik_step(setpoint, hand_quat, ee_p, ee_q, open_)
        if args.debug:
            err = torch.linalg.norm(goal - ee_pose()[0][0]).item()
            cube_p = cube.data.root_pos_w.torch[0].cpu().numpy().round(3)
            finger_q = robot.data.joint_pos.torch[0, 7:].cpu().numpy().round(3)
            print(f"    phase goal={goal.cpu().numpy().round(3)} ee_err={err:.3f} cube={cube_p} fingers={finger_q}")
    success = cube.data.root_pos_w.torch[0, 2].item() > CUBE_SIZE / 2 + 0.10
    return steps, qa, success


# ----------------------------------------------------------------------------- main loop
os.makedirs(f"{args.out}/images", exist_ok=True)
f_act = open(f"{args.out}/actions.jsonl", "w")
f_qa = open(f"{args.out}/qa.jsonl", "w")
saved = tried = n_steps = n_qa = 0
max_tries = args.max_tries or 3 * args.episodes
while saved < args.episodes and tried < max_tries and app.is_running():
    tried += 1
    color = str(rng.choice(list(COLORS)))
    instruction = str(rng.choice(INSTRUCTIONS)).format(c=color)
    steps, qa, success = run_episode(color, first_image=f"images/ep{saved:04d}_t000.png")
    print(f"[try {tried:3d}] {instruction:35s} steps={len(steps):3d}  success={success}", flush=True)
    if args.debug:
        os.makedirs(f"{args.out}/debug", exist_ok=True)
        iio.imwrite(f"{args.out}/debug/try{tried:03d}.gif", np.stack([s[0] for s in steps]), duration=50, loop=0)
    if not success:
        continue
    for t, (rgb, action) in enumerate(steps):
        img = f"images/ep{saved:04d}_t{t:03d}.png"
        iio.imwrite(f"{args.out}/{img}", rgb)
        rec = {
            "episode": saved, "step": t, "image": img, "instruction": instruction,
            "action": [round(float(a), 5) for a in action], "action_tokens": " ".join(map(str, to_tokens(action))),
            "is_last": t == len(steps) - 1,
        }  # fmt: skip
        f_act.write(json.dumps(rec) + "\n")
    for q in qa:
        f_qa.write(json.dumps({"episode": saved, **q}) + "\n")
        n_qa += 1
    n_steps += len(steps)
    saved += 1
f_act.close()
f_qa.close()

meta = {
    "episodes": saved, "attempts": tried, "steps": n_steps, "qa_pairs": n_qa, "control_hz": 1 / (SIM_DT * DECIMATION),
    "image_size": args.img_size, "camera": {"eye": CAM_EYE, "lookat": CAM_LOOKAT},
    "action": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper(+1 open,-1 close)"],
    "action_tokens": {"bins": 256, "pos_range_m": POS_BIN_RANGE, "rot_range_rad": float(np.pi), "gripper_range": 1.0},
}  # fmt: skip
with open(f"{args.out}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"saved {saved}/{tried} episodes, {n_steps} steps, {n_qa} QA pairs -> {args.out}")
app.close()
