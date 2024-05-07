import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from typing import Dict


from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.utils.utils import load_config
from Humanoid_MARL.training.train_ant import train as train_ant
from Humanoid_MARL.training.train_ant_mappo import train as train_ant_mappo

gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def cmd_args():
    parser = argparse.ArgumentParser(description="Train PPO")
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        default="ants",
        help="environment name",
    )
    parser.add_argument(
        "-a",
        "--algo",
        type=str,
        default="mappo",
    )
    args = parser.parse_args()
    return args


def main(args):
    print(f"======Using GPU {gpu_index}======")
    # ================ Config ================
    config = load_config(str(args.env_name), str(args.algo))
    env_name = config["env_name"]
    project_name = f"MARL_ppo_{env_name}"
    # ================ Config ================
    gpu_info = f"GPU {gpu_index} | env_name {env_name}"
    logger = WandbLogger(project_name, config=config, notes=gpu_info)
    config["train_config"]["logger"] = logger
    if config["env_name"]:
        if args.algo == "ippo":
            train_ant(
                **config["train_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif args.algo == "mappo":
            train_ant_mappo(
                **config["train_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )


if __name__ == "__main__":
    args = cmd_args()
    main(args)
