from typing import Dict, List
import os
import torch
from datetime import datetime
from Humanoid_MARL.agent.ppo.agent import Agent

SAVE_MODELS = "./models"
SAVE_MODEL_NOTEBOOK = "../models"

def save_models(agents : List[Agent], network_arch : Dict, model_name : str, notebook : bool = False) -> None:
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
            f"num_steps_{i}": agent.num_steps
        }
        state_dicts["agents"].append(agent_dict)

    torch.save(state_dicts, filename)
    print(f"Agent Weights {agent.state_dict()}")
    print(f"Agents saved to {filename}")

def load_models(filename : str, agent_class : Agent, device : str = 'cuda') -> List[Agent]:
    state_dicts = torch.load(filename)
    network_arch = state_dicts["network_arch"]
    agents = []
    
    for agent_dict in state_dicts["agents"]:
        index = agent_dict["index"]
        agent = agent_class(**network_arch).to(device)
        agent.policy.load_state_dict(agent_dict[f"agent_policy_{index}"])
        agent.value.load_state_dict(agent_dict[f"agent_value_{index}"])
        agent.running_mean = agent_dict[f"running_mean_{index}"]
        agent.running_variance = agent_dict[f"running_variance_{index}"]
        agent.num_steps = agent_dict[f"num_steps_{index}"]
        agent.eval()
        agents.append(agent)

    print(f"Models loaded from {filename}")
    return agents