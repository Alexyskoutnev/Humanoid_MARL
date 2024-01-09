# Humanoid_MARL

## Overview

Humanoid_MARL (Multi-Agent Reinforcement Learning for Humanoid Robots) is a research project that explores the application of reinforcement learning, machine learning, and game thoery techniques to enable multiple humanoid robots to perform complex tasks in a multi-agent environment.

## Setup

To maintain consistency across systems, a Conda environment is used to manage all Python modules and dependencies in one spot.

### Conda Setup

1. **Install Conda:**
   If you don't have Conda installed, download and install Miniconda or Anaconda from the official website:
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

   If you want a fresh start without any prebuilt configuration do steps 2 - 4. However to use install previous build enviroment used in this project go to step 5.

2. **Open a Terminal/Command Prompt:**
   Open your terminal or command prompt to execute the following commands.

3. **Create Conda Environment:**
   ```bash
   conda create --name humanoid-marl python=3.11.5
   ```
4. **Activating Conda Enviroment:**
   ```bash
   conda activate humanoid-marl
   ```
5. **Install Previous Conda Enviroment:**
     ```bash
   conda env create -f environment.yml
   ```
   
## NOTICE
This repo in the development state, feel free to contact Alexy with any bug reports or additions needed to the repository. 
