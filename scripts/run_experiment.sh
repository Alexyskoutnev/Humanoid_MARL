#!/bin/bash

LOG_OUTPUT_DIR="log"

if [ -d "$LOG_OUTPUT_DIR" ]; then
    echo "Experiment logs are in /log directory"
else
    mkdir "$LOG_OUTPUT_DIR"
fi

# Default values
NUM_EXPERIMENTS=1
GPU_INDEX="0"

if [ "$#" -gt 0 ]; then
    GPU_INDEX="$1"
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Specify order based on PCI bus ID

LOG_NAME="log/output_"

echo -e "\e[32mExperiment ${GPU_INDEX}\e[0m"
LOG_NAME="log/output_1_experiment_1_gpu_${GPU_INDEX}.log"

# If GPU_INDEX is not provided, use the default GPU assignment
if [ -z "$GPU_INDEX" ]; then
    GPU_INDEX="0"
fi

echo "Using GPU $GPU_INDEX"

# Run the experiment with the specified GPU index
CUDA_VISIBLE_DEVICES="$GPU_INDEX" nohup python3 scripts/train_ppo.py > "$LOG_NAME" 2>&1 &