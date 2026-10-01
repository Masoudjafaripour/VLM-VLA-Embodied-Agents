"""Minimal Isaac Lab sim: a cube dropping onto a ground plane.

Run:  python rob_envs/isaac/minimal_sim.py            (headless, default)
      python rob_envs/isaac/minimal_sim.py --viz kit  (GUI)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app  # must launch before other isaaclab imports

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.3])

cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/ground", cfg)
cfg = sim_utils.DomeLightCfg(intensity=2000.0)
cfg.func("/World/light", cfg)

cube = RigidObject(
    RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
)

sim.reset()
for i in range(args.steps):
    if not app.is_running():
        break
    sim.step()
    cube.update(sim.get_physics_dt())
    if i % 50 == 0:
        print(f"step {i:4d}  cube z = {cube.data.root_pos_w[0, 2].item():.3f}")

app.close()
