#!/bin/bash

LOG_OUTPUT_DIR="log"

if [ -d "$LOG_OUTPUT_DIR" ]; then
    echo "Experiment logs are in /log directory"
else
    mkdir "$LOG_OUTPUT_DIR"
fi

if [ "$#" -gt 0 ]; then
    for arg in "$@"; do
        case "$arg" in
            num=*)
                NUM_EXPERIMENTS="${arg#*=}"
                ;;
        esac
    done
else
    NUM_EXPERIMENTS=1
fi

if [ "$NUM_EXPERIMENTS" -gt 10 ]; then
    echo "Don't not run more than 10 experiments at once"
    exit 1
fi

LOG_NAME="log/output_"

for ((i=1; i<=NUM_EXPERIMENTS; i++)); do
    echo "Experiment $i"
    LOG_NAME="log/output_${NUM_EXPERIMENTS}_experiment_${i}.log"
    nohup python3 scripts/train_ppo.py > "$LOG_NAME" 2>&1 &
done
