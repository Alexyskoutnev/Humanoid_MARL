#!/bin/bash

# Function to display help
display_help() {
    echo "Usage: $0 [-h] [GPU_INDEX]"
    echo "  -h: Display this help message"
    echo "  GPU_INDEX: Optional GPU index (default: 0)"
}

# Check for help flag
if [ "$1" == "-h" ]; then
    display_help
    exit 0
fi

# Set GPU_INDEX
if [ "$#" -gt 0 ]; then
    GPU_INDEX="$1"
else 
    GPU_INDEX="0 1"  # Set GPU_INDEX to "0 1" when no argument is provided
fi

echo -e "\e[31mOption to clear logs and kill running environments\e[0m"
./scripts/clear_logs.sh
./scripts/kills_runs.sh

echo "Running new experiments on GPU indexes: $GPU_INDEX"

# Loop over GPU indexes
for index in $GPU_INDEX; do
    echo "Running experiment on GPU index $index"
    ./scripts/run_experiment.sh $index
    sleep 5
done

nvidia-smi
