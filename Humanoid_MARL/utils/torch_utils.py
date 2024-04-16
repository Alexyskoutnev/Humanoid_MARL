from typing import Dict, List, Optional, Union, Tuple
import os
import torch
from datetime import datetime
from Humanoid_MARL.agent.ppo.agent import Agent

SAVE_MODELS = "./models"
SAVE_MODEL_NOTEBOOK = "../models"


def save_model(agents: List[Agent], network_arch: Dict, model_name: str) -> None:
    filename = os.path.join(SAVE_MODELS, model_name)
    state_dicts = {"network_arch": network_arch, "agents": []}
    for i, agent in enumerate(agents):
        agent_dict = {
            "index": i,
            f"agent_policy_{i}": agent.policy.state_dict(),
            f"agent_value_{i}": agent.value.state_dict(),
            f"running_mean_{i}": agent.running_mean,
            f"running_variance_{i}": agent.running_variance,
            f"num_steps_{i}": agent.num_steps,
        }
        state_dicts["agents"].append(agent_dict)

    torch.save(state_dicts, filename)
    print(f"Agents saved to {filename}")


def save_models(
    agents: List[Agent], network_arch: Dict, model_name: str, notebook: bool = False
) -> None:
    if not notebook:
        filename = os.path.join(SAVE_MODELS, model_name)
    elif notebook:
        filename = os.path.join(SAVE_MODEL_NOTEBOOK, model_name)

    state_dicts = {"network_arch": network_arch, "agents": []}

    for i, agent in enumerate(agents):
        agent_dict = {
            "index": i,
            f"agent_policy_{i}": agent.policy.state_dict(),
            f"agent_value_{i}": agent.value.state_dict(),
            f"running_mean_{i}": agent.running_mean,
            f"running_variance_{i}": agent.running_variance,
            f"num_steps_{i}": agent.num_steps,
        }
        state_dicts["agents"].append(agent_dict)

    torch.save(state_dicts, filename)
    print(f"Agent Weights {agent.state_dict()}")
    print(f"Agents saved to {filename}")


def load_models_empty(
    filename: Union[str, None],
    agent_class: Agent,
    device: str = "cpu",
    network_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
    num_agents: Optional[int] = 2,
) -> List[Agent]:
    network_arch = {
        "policy_layers": network_config["POLICY_LAYERS"],
        "value_layers": network_config["VALUE_LAYERS"],
        "entropy_cost": training_config["entropy_cost"],
        "discounting": training_config["discounting"],
        "reward_scaling": training_config["reward_scaling"],
        "device": training_config["device"],
        "network_config": network_config,
    }
    agents = []
    for i in range(num_agents):
        agent = agent_class(**network_arch).to(device)
        agent.eval()
        agents.append(agent)

    return agents


def load_models(filename: str, agent_class: Agent, device: str = "cpu") -> List[Agent]:

    state_dicts = torch.load(filename)
    network_arch = state_dicts["network_arch"]
    num_agents = len(state_dicts["agents"])
    agents = [None for _ in range(num_agents)]

    for agent_dict in state_dicts["agents"]:
        index = agent_dict["index"]
        agent = agent_class(**network_arch).to(device)
        agent.policy.load_state_dict(agent_dict[f"agent_policy_{index}"])
        agent.value.load_state_dict(agent_dict[f"agent_value_{index}"])
        agent.running_mean = agent_dict[f"running_mean_{index}"]
        agent.running_variance = agent_dict[f"running_variance_{index}"]
        agent.num_steps = agent_dict[f"num_steps_{index}"]
        agent.eval()
        agents[index] = agent
    if len(agents) == 0:
        raise ValueError("No agents loaded")
    if any([a is None for a in agents]):
        raise ValueError("Some agents failed to load")
    print(f"Models loaded from {filename}")
    return agents


def _nan_filter(*arr: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    _arr_return = []
    for a in arr[0]:
        if isinstance(a, torch.Tensor):
            nan_mask = torch.isnan(a)
            if torch.any(nan_mask):
                a = torch.where(nan_mask, torch.tensor(0.1), a)
        _arr_return.append(a)
    return tuple(_arr_return)
