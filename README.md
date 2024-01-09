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

# Experiements 
In this research project, we conduct experiments to evaluate potential ideas and research directions. Each experiment is initially performed in a Jupyter notebook to assess the feasibility and potential of a particular idea before building the final framework. This iterative process allows us to gauge the experimental idea's potential and determine how it contributes to the overarching research goals.
## Format
- Each experiment is designed to accomplish a task within a timeframe of '1-3 days'. The emphasis on shorter tasks promotes efficiency and enables rapid iteration. 
- At the beginning of each Jupyter notebook, a comprehensive description is provided to outline the experiment's objectives. This narrative serves to contextualize the experiment and align it with the broader research goals.
## Structure 
Experiments are organized within the `experiments` folder. This structure facilitates an iterative exploration of various ideas, allowing us to gather insights before defining the framework's scope. The goal is to first conduct experiments, formulate valid and novel research questions, and then guide the subsequent development of the research project.

## NOTICE
This repo in the development state, feel free to contact Alexy with any bug reports or additions needed to the repository. 
