import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Dict

from Humanoid_MARL.agent.ppo.train_torch import train
from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.utils.utils import load_reward_config
from Humanoid_MARL import CONFIG_TRAIN, CONFIG_REWARD, CONFIG_AGENT

gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def debug_config(config: Dict) -> Dict:
    update_config = {
        "unroll_length": 1,
        "num_minibatches": 1,
        "num_envs": 1,
        "batch_size": 1,
        "unroll_length": 1,
    }
    return config.update(update_config)


def main():
    print(f"======Using GPU {gpu_index}======")
    # ================ Config ================
    with open(CONFIG_TRAIN, "r") as f:
        config = yaml.safe_load(f)
        if config["debug"]:
            debug_config(config)
            env_name = config["env_name"] + "_debug"
        else:
            env_name = config["env_name"]
    project_name = f"MARL_ppo_{env_name}"
    env_config = load_reward_config(CONFIG_REWARD, env_name)
    with open(CONFIG_AGENT, "r") as f:
        agent_config = yaml.safe_load(f)
    # ================ Config ================
    # ================ Logging ===============
    if not config["debug"]:
        gpu_info = f"GPU {gpu_index} | env_name {env_name}"
        logger = WandbLogger(project_name, config=config, notes=gpu_info)
        config["logger"] = logger
    # ================ Timing ================
    times = [datetime.now()]
    # ================ Timing ================
    if config["debug"]:
        train(**config, env_config=env_config)
    else:
        train(**config, env_config=env_config, agent_config=agent_config)
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")


if __name__ == "__main__":
    main()
