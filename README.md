# UniFP Learning

Personal research and development repository for B2Z1 control, force estimation, and policy adaptation experiments.

## Overview

This repository is used for development and experiments around B2Z1 whole-body control in simulation. Current work focuses on force-position control, trajectory tracking, and estimator-policy decoupling for later controller replacement.

The main goal is to separate the estimation module from the reinforcement learning policy, so that the estimator can be reused by other downstream controllers or tasks.

## Current Focus

- B2Z1 position-force control in Isaac Gym
- External force related observation and estimation
- Decoupling estimator and actor
- Trajectory tracking experiments
- Interface preparation for future controller replacement

## Environment

Recommended environment:

- Ubuntu 20.04 or 22.04
- Python 3.8
- CUDA-compatible GPU
- Isaac Gym Preview 4

Example setup:

```bash
conda create -n unifp python=3.8
conda activate unifp
pip install numpy matplotlib wandb
```

Isaac Gym should be installed separately under the local workspace or environment.

## Repository Structure

- `legged_gym/envs/b2/`
	- B2Z1 task configuration and environment implementation
- `legged_gym/b2_gym_learn/ppo_cse_pf/`
	- PPO training code, actor-critic structure, and adaptation modules
- `legged_gym/scripts/`
	- Training, evaluation, and experiment scripts
- `resources/robots/b2z1/`
	- Robot description files and related assets

## Common Workflows

### Training

```bash
cd legged_gym/scripts
python train_b2z1posforce.py --task=b2z1_pos_force --headless
```

### Evaluation

```bash
cd legged_gym/scripts
python play_b2z1posforce.py --task=b2z1_pos_force --load_run=<run_name>
```

## Current Research Notes

The current development direction includes:

- checking which observation terms are actually informative for external force estimation
- restructuring the adaptation / estimator module into a reusable component
- reducing coupling between latent estimation and policy optimization
- preparing for future replacement of the reinforcement learning actor with other control modules

## Notes

This repository is intended for personal and internal research use. Some scripts, logs, and local experiment artifacts may change frequently during development.

## Upstream

This repository is based on an upstream UniFP codebase and has been modified for local research and development.

