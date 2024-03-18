from typing import Dict

import random
import numpy as np
import os
import torch
import yaml
from Humanoid_MARL import CONFIG_TRAIN, CONFIG_REWARD, CONFIG_AGENT, CONFIG_NETWORK


def _debug_config(config: Dict) -> Dict:
    update_config = {
        "unroll_length": 8,
        "num_minibatches": 2,
        "num_envs": 1,
        "batch_size": 1,
        "unroll_length": 2,
    }
    return config.update(update_config)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_agent_config(path: str) -> Dict:
    with open(CONFIG_AGENT, "r") as f:
        return yaml.safe_load(f)


def load_network_config(path: str) -> Dict:
    with open(CONFIG_NETWORK, "r") as f:
        return yaml.safe_load(f)


def load_reward_config(path: str, env: str) -> Dict:
    if env in ["humanoid", "humanoid_debug"]:
        path = os.path.join(CONFIG_REWARD, "reward_humanoid.yaml")
    elif env in [
        "humanoids",
        "humanoids_debug",
        "humanoids_wall",
        "humanoids_wall_debug",
    ]:
        path = os.path.join(CONFIG_REWARD, "reward_humanoids.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_config(path: str) -> Dict:
    with open(CONFIG_TRAIN, "r") as f:
        config = yaml.safe_load(f)
        if config["debug"]:
            _debug_config(config)
            env_name = config["env_name"] + "_debug"
        else:
            env_name = config["env_name"]
        config["env_name"] = env_name
        return config


def load_config() -> Dict:
    train_config = load_train_config(CONFIG_TRAIN)
    env_config = load_reward_config(CONFIG_REWARD, train_config["env_name"])
    agent_config = load_agent_config(CONFIG_AGENT)
    network_config = load_network_config(CONFIG_NETWORK)
    return {
        "env_name": train_config["env_name"],
        "env_config": env_config,
        "agent_config": agent_config,
        "network_config": network_config,
        "train_config": train_config,
    }
