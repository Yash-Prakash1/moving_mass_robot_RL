# Moving Center of Mass Robot, Reinforcement Learning

Training a moving center of mass robot to traverse uneven, unseen terrain at high
speed using reinforcement learning and a single onboard camera.

## Thesis

This repository backs my research thesis at the Biorobotics Lab, Carnegie Mellon
University. You can read the full write up here, [Thesis.pdf](Thesis.pdf).

## Overview

This is the thesis stage of the VeRT robot project. A wheeled robot with an actuated,
shifting mass learns to keep its balance and drive quickly across rough ground it has
never seen before. Control policies are trained with Proximal Policy Optimization
(PPO) in massively parallel simulation, and terrain understanding comes from a single
camera processed by a convolutional network rather than from privileged ground truth.

An earlier prototype of the same robot, built in the CoppeliaSim simulator, lives in
[VeRT_PPO](https://github.com/Yash-Prakash1/VeRT_PPO). This repository is the later,
more capable continuation.

## What it does

* Trains a PPO policy on the robot defined in `assets/robot_mvw.urdf`.
* Runs many environments in parallel in NVIDIA Isaac Gym, with a MuJoCo backend
  available as an alternative.
* Generates procedural terrain including slopes, stairs, waves, discrete obstacles,
  pyramids, and stepping stones, so the policy learns to generalize rather than
  memorize one map.
* Reads a single onboard camera through a convolutional network to perceive the
  terrain ahead, instead of relying on a full state of the world.
* Shapes behavior with a reward that favors forward speed toward the goal, applies a
  small torque penalty for efficiency, tracks body position, and adds a bonus for
  reaching the target. An episode ends when the body falls below a height threshold.

## Stack

* Python, PyTorch
* NVIDIA Isaac Gym for parallel GPU simulation
* MuJoCo as an alternative physics backend
* Weights & Biases for experiment tracking
* PPO implemented from scratch in the SpinningUp style (`core.py`, `ppo.py`,
  `ppo2.py`, `ppo3.py`)

## Results

The policy was trained and run in simulation, and the trained checkpoints and
experiment records are included as a record of that work.

The images below are real onboard camera views from the simulated robot, the same
single camera input the convolutional network learns terrain from.

| | |
|---|---|
| ![Onboard camera view 1](results/camera_views/rgb_env0_cam0.png) | ![Onboard camera view 2](results/camera_views/rgb_env10_cam0.png) |
| ![Onboard camera view 3](results/camera_views/rgb_env20_cam0.png) | ![Onboard camera view 4](results/camera_views/rgb_env30_cam0.png) |

* `checkpoints/`, the trained policy and value networks, including the thesis runs and
  the camera based CNN models.
* `experiments/`, archived Weights & Biases run records for the training runs.
* `results/camera_views/`, onboard camera frames captured from the simulation.

The full method and quantitative results are in [Thesis.pdf](Thesis.pdf).

## Repository layout

* `scripts/main.py`, the training entry point
* `scripts/multiple_env_make.py`, the parallel Isaac Gym environment
* `scripts/mujoco_env_make.py`, the MuJoCo environment
* `scripts/terrain_utils_update.py`, procedural terrain generation
* `scripts/core.py`, the actor critic networks and PPO buffer
* `scripts/ppo*.py`, the PPO training loops, including the camera CNN path
* `assets/`, the robot URDF models
* `checkpoints/`, saved policies, including the thesis runs
* `experiments/`, archived training run records
* `results/camera_views/`, sample onboard camera frames

## How to run

This project depends on NVIDIA Isaac Gym, which you install separately from NVIDIA
after agreeing to their license. A CUDA capable GPU is required for the Isaac Gym
path.

```bash
# 1. Install Isaac Gym from NVIDIA, then its Python bindings
# 2. Install the Python dependencies
pip install torch numpy matplotlib wandb torchvision

# 3. Launch training
cd scripts
python main.py
```

Training progress, rewards, and losses are logged to Weights & Biases. Trained
policies are saved as checkpoints. Set your W&B key with the `WANDB_API_KEY`
environment variable before training.

## Notes

This is research code from an active thesis project, so several environment variants
and PPO iterations are kept side by side. `multiple_env_make.py` with `ppo2.py` is
the main path used for the thesis results.
