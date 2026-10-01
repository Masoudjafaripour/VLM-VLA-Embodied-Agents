"""Minimal Isaac Lab robot sim: a Franka Panda arm waving its joints sinusoidally.

Run:  python rob_envs/isaac/franka_sim.py --viz kit             (GUI)
      python rob_envs/isaac/franka_sim.py                       (headless)
      python rob_envs/isaac/franka_sim.py --record --steps 800  (save GIF + joint plot to outputs/isaac/)
"""
import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100000)
parser.add_argument("--record", action="store_true", help="save a GIF and a joint-angle plot")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.record:
    args.enable_cameras = True
app = AppLauncher(args).app  # must launch before other isaaclab imports

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

OUT_DIR = "outputs/isaac"
EYE, LOOKAT = [2.0, 2.0, 1.5], [0.0, 0.0, 0.4]

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
sim.set_camera_view(EYE, LOOKAT)

cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)
cfg = sim_utils.DomeLightCfg(intensity=2000.0)
cfg.func("/World/light", cfg)

robot = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Franka"))

camera = None
if args.record:
    camera = Camera(
        CameraCfg(
            prim_path="/World/RecCam",
            width=400,
            height=300,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=40.0),
        )
    )

sim.reset()
default_q = robot.data.default_joint_pos.torch.clone()  # (1, 9): 7 arm + 2 finger joints
robot.write_joint_position_to_sim_index(position=default_q)  # start in the default pose
if camera is not None:
    camera.set_world_poses_from_view(torch.tensor([EYE], device=sim.device), torch.tensor([LOOKAT], device=sim.device))
print("joints:", robot.joint_names)

dt = sim.get_physics_dt()
frames, ts, qs = [], [], []
for i in range(args.steps):
    if not app.is_running():
        break
    t = i * dt
    target = default_q.clone()
    target[:, :7] += 0.5 * math.sin(2 * math.pi * 0.25 * t)       # swing arm joints
    target[:, 7:] = 0.02 + 0.02 * math.sin(2 * math.pi * 0.5 * t)  # open/close gripper
    robot.actuators.target_command.set_position_index(value=target)
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)
    ts.append(t)
    qs.append(robot.data.joint_pos.torch[0, :7].cpu().numpy().copy())
    if camera is not None and i % 4 == 0:  # 25 fps
        camera.update(dt)
        frames.append(camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().copy())
    if i % 100 == 0:
        print(f"t={t:5.2f}s  q={qs[-1].round(2)}")

if args.record:
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(OUT_DIR, exist_ok=True)
    imageio.mimsave(f"{OUT_DIR}/franka.gif", frames[1:], duration=40, loop=0)  # skip first (blank) frame

    qs = np.stack(qs)
    plt.figure(figsize=(8, 4))
    for j in range(7):
        plt.plot(ts, qs[:, j], label=f"joint{j + 1}")
    plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    plt.title("Franka joint positions")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/franka_joints.png", dpi=120)
    print(f"saved {OUT_DIR}/franka.gif ({len(frames) - 1} frames) and {OUT_DIR}/franka_joints.png")

app.close()
