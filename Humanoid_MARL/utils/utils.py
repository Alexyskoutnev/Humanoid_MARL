from typing import Dict

import random
import numpy as np
import os
import torch
import yaml
from Humanoid_MARL import *
from Humanoid_MARL.agent.ppo.agent import Agent


def get_grad_info(agent: Agent) -> Dict[str, float]:
    gradient_list_policy = []
    gradient_list_value = []
    for param in agent.policy.parameters():
        if param.grad is not None:
            gradient_list_policy.append(param.grad.cpu().detach().numpy())
    for param in agent.value.parameters():
        if param.grad is not None:
            gradient_list_value.append(param.grad.cpu().detach().numpy())

    _policy_gradients = np.concatenate(
        [gradient.flatten() for gradient in gradient_list_policy]
    )
    mean_gradient_policy = float(np.mean(_policy_gradients))
    std_gradient_policy = float(np.std(_policy_gradients))
    max_gradient_policy = float(np.max(_policy_gradients))
    _value_gradients = np.concatenate(
        [gradient.flatten() for gradient in gradient_list_value]
    )
    mean_gradient_value = float(np.mean(_value_gradients))
    std_gradient_value = float(np.std(_value_gradients))
    max_gradient_value = float(np.max(_value_gradients))

    return {
        "mean_policy": mean_gradient_policy,
        "std_policy": std_gradient_policy,
        "max_policy": max_gradient_policy,
        "mean_value": mean_gradient_value,
        "std_value": std_gradient_value,
        "max_value": max_gradient_value,
    }


def _debug_config(config: Dict) -> Dict:
    update_config = {
        "unroll_length": 2,
        "num_minibatches": 4,
        "num_envs": 2,
        "batch_size": 64,
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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_network_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_reward_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_train_config(path: str) -> Dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        env_name = config["env_name"]
        config["env_name"] = env_name
        return config


def load_config(env_name: str = "humanoids", algo: str = "ippo") -> Dict:
    if env_name in ["humanoid", "humanoid_debug", "humanoid_wall"]:
        train_config = load_train_config(CONFIG_TRAIN_HUMANOID)
        env_config = load_reward_config(CONFIG_REWARD_HUMANOID)
        agent_config = load_agent_config(CONFIG_AGENT_HUMANOID)
        network_config = load_network_config(CONFIG_NETWORK_HUMANOID)
        return {
            "env_name": train_config["env_name"],
            "env_config": env_config,
            "agent_config": agent_config,
            "network_config": network_config,
            "train_config": train_config,
        }
    elif env_name in ["humanoids"]:
        train_config = load_train_config(CONFIG_TRAIN_HUMANOIDS)
        env_config = load_reward_config(CONFIG_REWARD_HUMANOIDS)
        agent_config = load_agent_config(CONFIG_AGENT_HUMANOIDS)
        network_config = load_network_config(CONFIG_NETWORK_HUMANOIDS)
        return {
            "env_name": train_config["env_name"],
            "env_config": env_config,
            "agent_config": agent_config,
            "network_config": network_config,
            "train_config": train_config,
        }
    elif env_name in ["ants"]:
        if algo == "ippo":
            train_config = load_train_config(CONFIG_TRAIN_ANT)
            env_config = load_reward_config(CONFIG_REWARD_ANT)
            agent_config = load_agent_config(CONFIG_AGENT_ANT)
            network_config = load_network_config(CONFIG_NETWORK_ANT)
            return {
                "env_name": train_config["env_name"],
                "env_config": env_config,
                "agent_config": agent_config,
                "network_config": network_config,
                "train_config": train_config,
            }
        elif algo == "mappo":
            train_config = load_train_config(CONFIG_TRAIN_ANT_MAPPO)
            env_config = load_reward_config(CONFIG_REWARD_ANT_MAPPO)
            agent_config = load_agent_config(CONFIG_AGENT_ANT_MAPPO)
            network_config = load_network_config(CONFIG_NETWORK_ANT_MAPPO)
            return {
                "env_name": train_config["env_name"],
                "env_config": env_config,
                "agent_config": agent_config,
                "network_config": network_config,
                "train_config": train_config,
            }

        elif algo == "isac":
            pass

        elif algo == "masac":
            pass

        elif algo == "maddpg":
            pass

        elif algo == "iddpg":
            pass

        else:
            train_config = load_train_config(CONFIG_TRAIN_ANT)
            env_config = load_reward_config(CONFIG_REWARD_ANT)
            agent_config = load_agent_config(CONFIG_AGENT_ANT)
            network_config = load_network_config(CONFIG_NETWORK_ANT)
            return {
                "env_name": train_config["env_name"],
                "env_config": env_config,
                "agent_config": agent_config,
                "network_config": network_config,
                "train_config": train_config,
            }
    elif env_name in ["point_mass"]:
        train_config = load_train_config(CONFIG_TRAIN_POINT_MASS)
        env_config = load_reward_config(CONFIG_REWARD_POINT_MASS)
        agent_config = load_agent_config(CONFIG_AGENT_POINT_MASS)
        network_config = load_network_config(CONFIG_NETWORK_POINT_MASS)
        return {
            "env_name": train_config["env_name"],
            "env_config": env_config,
            "agent_config": agent_config,
            "network_config": network_config,
            "train_config": train_config,
        }
    elif env_name in ["linked_balls"]:
        train_config = load_train_config(CONFIG_TRAIN_LINKED_BALLS)
        env_config = load_reward_config(CONFIG_REWARD_LINKED_BALLS)
        agent_config = load_agent_config(CONFIG_AGENT_LINKED_BALLS)
        network_config = load_network_config(CONFIG_NETWORK_LINKED_BALLS)
        return {
            "env_name": train_config["env_name"],
            "env_config": env_config,
            "agent_config": agent_config,
            "network_config": network_config,
            "train_config": train_config,
        }
    elif env_name in ["simple_robots"]:
        train_config = load_train_config(CONFIG_TRAIN_SIMPLE_ROBOT)
        env_config = load_reward_config(CONFIG_REWARD_SIMPLE_ROBOT)
        agent_config = load_agent_config(CONFIG_AGENT_SIMPLE_ROBOT)
        network_config = load_network_config(CONFIG_NETWORK_SIMPLE_ROBOT)
        return {
            "env_name": train_config["env_name"],
            "env_config": env_config,
            "agent_config": agent_config,
            "network_config": network_config,
            "train_config": train_config,
        }
