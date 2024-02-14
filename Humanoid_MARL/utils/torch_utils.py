from typing import Dict
import os
import torch
from datetime import datetime
from Humanoid_MARL.agent.ppo.agent import Agent

SAVE_MODELS = "./models"
SAVE_MODEL_NOTEBOOK = "../models"

def save_models(agents : list[Agent], network_arch : Dict, model_name : str, notebook : bool = False):
    if not notebook:
        filename = os.path.join(SAVE_MODELS, model_name)
    elif notebook:
        filename = os.path.join(SAVE_MODEL_NOTEBOOK, model_name)
    state_dicts = {"network_arch": network_arch}
    for i, agent in enumerate(agents):
        state_dicts[f'agent_{i}'] = agent.state_dict()
    torch.save(state_dicts, filename)
    print(f"Agent Weights {agent.state_dict()}")
    print(f"Agents saved to {filename}")

def load_models(filename, agent_class, device: str = 'cuda'):
    state_dicts = torch.load(filename)
    network_arch = state_dicts["network_arch"]
    agents = []
    for i in range(len(state_dicts) - 1):  # Subtract 1 to exclude the network_arch entry
        agent = agent_class(**network_arch).to(device)
        agent.load_state_dict(state_dicts[f'agent_{i}'])
        agent.eval()
        agents.append(agent)

    print(f"Models loaded from {filename}")
    return agents