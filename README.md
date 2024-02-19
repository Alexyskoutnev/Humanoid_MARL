# Humanoid_MARL

## Overview

Humanoid_MARL (Multi-Agent Reinforcement Learning for Humanoid Robots) is a research project that explores the application of reinforcement learning, machine learning, and game thoery techniques to enable multiple humanoid robots to perform complex tasks in a multi-agent environment.

## Setup

To maintain consistency across systems, a Conda environment is used to manage all Python modules and dependencies in one spot.
A installation script is provided in this repo and can be evoked by running the following command

## Installing Humanoid Enviroment
```bash
./install_env.sh
```

Once all the repository dependencies are installed, activate the Conda environment by running:
```bash
conda activate humanoids
```

## Training Humanoid Agents
A built-in script is provided for training the humanoids. To execute the bash script, use the following command:
```
./scripts/run.sh
```
This script will terminate all running trainers and restart the logging files within the `log` directory.
