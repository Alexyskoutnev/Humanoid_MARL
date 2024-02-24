import torch
import os


def copy(file_path: str, num_agents: int = 2):
    model_dict = torch.load(file_path)
    single_agent = model_dict["agents"][0]
    state_dicts = {"network_arch": model_dict["network_arch"], "agents": []}
    for i in range(num_agents):
        agent_dict = {
            "index": i,
            f"agent_policy_{i}": single_agent["agent_policy_0"],
            f"agent_value_{i}": single_agent["agent_value_0"],
            f"running_mean_{i}": single_agent["running_mean_0"],
            f"running_variance_{i}": single_agent["running_variance_0"],
            f"num_steps_{i}": single_agent["num_steps_0"],
        }
        state_dicts["agents"].append(agent_dict)
    name, pt = os.path.splitext(file_path)
    name += f"_copy_{num_agents}.pt"
    torch.save(state_dicts, name)


if __name__ == "__main__":
    single_agent_torch_pt = "20240226_092325_ppo_humanoid_standing.pt"
    model_file = os.path.join("models", single_agent_torch_pt)
    copy(model_file)
