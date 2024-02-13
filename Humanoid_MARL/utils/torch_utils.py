import os
import torch
from datetime import datetime

SAVE_MODELS = "./models"

def save_models(agents, network_arch, type="ppo", env="humanoid"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_name = f"{timestamp}_{type}_{env}.pt"
    filename = os.path.join(SAVE_MODELS, timestamped_name)
    state_dicts = {"network_arch": network_arch}
    for i, agent in enumerate(agents):
        state_dicts[f'agent_{i}'] = agent.state_dict()
    torch.save(state_dicts, filename)
    print(f"Agent Weights {agent.state_dict()}")
    print(f"Agents saved to {filename}")

def load_models(filename, agent_class, device: str = 'cpu'):
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