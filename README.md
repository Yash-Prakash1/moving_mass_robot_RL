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

## Repository layout

* `scripts/main.py`, the training entry point
* `scripts/multiple_env_make.py`, the parallel Isaac Gym environment
* `scripts/mujoco_env_make.py`, the MuJoCo environment
* `scripts/terrain_utils_update.py`, procedural terrain generation
* `scripts/core.py`, the actor critic networks and PPO buffer
* `scripts/ppo*.py`, the PPO training loops, including the camera CNN path
* `assets/`, the robot URDF models
* `scripts/Checkpoints/`, saved policies, including the thesis runs

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
policies are written to `scripts/Checkpoints/`.

## Notes

This is research code from an active thesis project, so several environment variants
and PPO iterations are kept side by side. `multiple_env_make.py` with `ppo2.py` is
the main path used for the thesis results.
