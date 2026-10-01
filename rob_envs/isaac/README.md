# Isaac Lab – minimal sims

Small, self-contained examples on **Isaac Sim 6.1 + Isaac Lab 3.0**.
- **Isaac Sim**: the NVIDIA simulator (PhysX physics, RTX rendering, sensors).
- **Isaac Lab**: Python framework on top of Isaac Sim for robot learning (RL/IL envs, parallel sims, robot configs).

| File | What it does |
|---|---|
| `minimal_sim.py` | Drops a red cube onto a ground plane; prints its height. Checks that the install works. |
| `franka_sim.py` | Franka Panda arm swings its 7 joints sinusoidally (0.25 Hz) and opens/closes the gripper. Optional GIF + joint plot. |
| `h1_walk.sh` | Unitree H1 humanoids walking with NVIDIA's **pretrained** RL policy (task `Isaac-Velocity-Flat-H1`, RSL-RL). |
| `h1_train.sh` | Trains H1 walking from scratch (PPO) and keeps a live plot of the learning curves. |
| `plot_training.py` | Plots RSL-RL training curves from TensorBoard logs into a PNG (matplotlib Agg), refreshing it periodically. |
| `vqa_data/` | Collects (image, instruction, action) + (image, question, answer) data with a scripted Franka pick expert. See [vqa_data/README.md](vqa_data/README.md). |

## Install (separate env, Python 3.12)
```bash
python3 -m venv isaac_venv && source isaac_venv/bin/activate
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com
pip install isaaclab --extra-index-url https://pypi.nvidia.com
pip install "moviepy>=1.0.3,<2.0.0.dev0" imageio-ffmpeg   # only for video recording
```
See the [Isaac Lab install guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) if the versions change.

## Run (from repo root, `isaac_venv` active)
Isaac Lab 3.0 runs **headless by default**; add `--viz kit` to open the GUI (there is no `--headless` flag).

```bash
# 1. sanity check
python rob_envs/isaac/minimal_sim.py
python rob_envs/isaac/minimal_sim.py --viz kit

# 2. Franka arm
python rob_envs/isaac/franka_sim.py --viz kit              # watch it
python rob_envs/isaac/franka_sim.py --record --steps 800   # -> outputs/isaac/franka.gif, franka_joints.png

# 3. H1 humanoid walking (pretrained policy, 4 robots)
bash rob_envs/isaac/h1_walk.sh            # GUI, runs until closed
bash rob_envs/isaac/h1_walk.sh --video    # headless, records a 300-frame mp4
```
The H1 checkpoint is downloaded to `rob_envs/isaac/.pretrained_checkpoints/` (gitignored), and `--video` writes to
`.../videos/play/clip_0000.mp4` inside it. A copy is kept at `outputs/isaac/h1_walk.mp4`.

**Train H1 walking yourself** (instead of the pretrained policy), with live learning curves:
```bash
bash rob_envs/isaac/h1_train.sh              # 4096 envs, default iterations
bash rob_envs/isaac/h1_train.sh 1024 300     # num_envs, max_iterations (~2.5 min on an RTX 3090)
cd rob_envs/isaac && isaaclab play --rl_library rsl_rl --task Isaac-Velocity-Flat-H1 --num_envs 4 --checkpoint latest --viz kit
```
`h1_train.sh` runs `isaaclab train` and, alongside it, `plot_training.py`, which reads the run's TensorBoard log and
re-saves `logs/rsl_rl/h1_flat/<run>/training_curves.png` every 15 s (matplotlib `Agg`, works headless). Open the PNG
in VS Code and it refreshes as training goes. Panels: mean reward, episode length, velocity tracking error/rewards,
losses, terminations (falls vs time-outs) and policy std. Example: `outputs/isaac/h1_training_curves.png`.

Plot any run on its own: `python rob_envs/isaac/plot_training.py [--run <run_dir>] [--once]`.
Checkpoints (`model_*.pt`) are saved in the same run dir (`logs/` is gitignored).
Other tasks: `Isaac-Velocity-Rough-H1`, `Isaac-Velocity-Flat-G1`, `Isaac-Velocity-Rough-G1`.

## Notes
- **First run is slow**: you'll be asked to accept the EULA (or set `OMNI_KIT_ACCEPT_EULA=YES`), and the first GUI launch compiles RTX shaders (several minutes, no output). To watch progress:
  `tail -f isaac_venv/lib/python3.12/site-packages/isaacsim/kit/logs/Kit/IsaacLab/3.0/kit_*.log`
- Robot models and checkpoints are fetched from NVIDIA's servers, so internet is required on first use.
- The startup table printed for the Franka lists per-joint stiffness/damping, position limits, and max velocity/effort.
