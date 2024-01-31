# RL Experiment Script

This Bash script sets up RL experiment logs and executes a Python script to start the RL training runner. The `num` argument determines the number of instances to train simultaneously, with a maximum limit of 10.

## Running the Script

- To initiate a single training runner:
```bash
./scripts/run_experiment.sh
```
- To initiate multiple training runners:
```bash
./scripts/run_experiment.sh num=5
```