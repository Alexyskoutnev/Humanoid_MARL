import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Dict

from Humanoid_MARL.agent.ppo.train_lstm import train
from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.utils.utils import load_config
from Humanoid_MARL import CONFIG_TRAIN, CONFIG_REWARD, CONFIG_AGENT, CONFIG_NETWORK

gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "1")


def main():
    print(f"======Using GPU {gpu_index}======")
    # ================ Config ================
    config = load_config()
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
        train(
            **config["train_config"],
            env_config=config["env_config"],
            network_config=config["network_config"],
            agent_config=config["agent_config"],
        )
    else:
        train(
            **config["train_config"],
            env_config=config["env_config"],
            agent_config=config["agent_config"],
            network_config=config["network_config"],
        )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


if __name__ == "__main__":
    main()
