#!/bin/bash

echo "Clearing logs and killing running enviroments"
./scripts/clear_logs.sh
./scripts/kills_runs.sh

echo "Running new experiments"
./scripts/run_experiment.sh
