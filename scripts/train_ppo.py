import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
from typing import Dict

from Humanoid_MARL.agent.ppo.train_humanoids import train as train_humanoid
from Humanoid_MARL.agent.ppo.train_lstm import train as train_lstm
from Humanoid_MARL.agent.ppo.train_ant import train as train_ant
from Humanoid_MARL.agent.ppo.linked_ball_train import train as train_linked_ball
from Humanoid_MARL.agent.ppo.train_point_mass import train as train_point_mass
from Humanoid_MARL.agent.ppo.train_simple_robot import train as train_simple_robot
from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.utils.utils import load_config
import Humanoid_MARL

gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "1")


def cmd_args():
    parser = argparse.ArgumentParser(description="Train PPO")
    parser.add_argument(
        # "-e", "--env_name", type=str, default="ants", help="environment name"
        "-e",
        "--env_name",
        type=str,
        # default="point_mass",
        default="ants",
        # default="humanoids",
        # default="linked_balls",
        # default="simple_robots",
        help="environment name",
    )
    args = parser.parse_args()
    return args


def main(args):
    print(f"======Using GPU {gpu_index}======")
    # ================ Config ================
    config = load_config(str(args.env_name))
    env_name = config["env_name"]
    project_name = f"MARL_ppo_{env_name}"
    # ================ Config ================
    # ================ Logging ===============
    if not config["train_config"]["debug"]:
        gpu_info = f"GPU {gpu_index} | env_name {env_name}"
        logger = WandbLogger(project_name, config=config, notes=gpu_info)
        config["train_config"]["logger"] = logger
    # ================ Timing ================
    times = [datetime.now()]
    # ================ Timing ================
    if config["train_config"]["debug"]:
        if (
            config["train_config"]["env_name"] == "humanoids"
            or config["train_config"]["env_name"] == "humanoids_wall"
        ):
            train_humanoid(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif (
            config["train_config"]["env_name"] == "ants"
            or config["train_config"]["env_name"] == "ants_debug"
        ):
            train_ant(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif config["train_config"]["env_name"] == "point_mass":
            train_point_mass(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif config["train_config"]["env_name"] == "linked_balls":
            train_linked_ball(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif (
            config["train_config"]["env_name"] == "simple_robots"
            or config["train_config"]["env_name"] == "simple_robots_debug"
        ):
            train_simple_robot(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )

    else:
        if (
            config["train_config"]["env_name"] == "humanoids"
            or config["train_config"]["env_name"] == "humanoids_wall"
        ):
            train_humanoid(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif config["train_config"]["env_name"] == "ants":
            train_ant(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif config["train_config"]["env_name"] == "point_mass":
            train_point_mass(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif config["train_config"]["env_name"] == "linked_balls":
            train_linked_ball(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )
        elif config["train_config"]["env_name"] == "simple_robots":
            train_simple_robot(
                **config["train_config"],
                env_config=config["env_config"],
                network_config=config["network_config"],
                agent_config=config["agent_config"],
            )


if __name__ == "__main__":
    args = cmd_args()
    main(args)
