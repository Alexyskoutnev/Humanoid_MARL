import matplotlib.pyplot as plt
import numpy as np
import torch
import jax
from Humanoid_MARL.agent.ppo.train_torch import Agent, get_agent_actions
from brax.envs.wrappers import torch as torch_wrapper
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
from Humanoid_MARL import envs
from Humanoid_MARL.utils.visual import save_rgb_image

def plot_rgb_image(rgb_array):
    """
    Plot an RGB image represented as a NumPy array.

    Parameters:
    - rgb_array: NumPy array representing the RGB image. It should have shape (height, width, 3).
    """
    if rgb_array.shape[2] != 3:
        raise ValueError("Input array should have shape (height, width, 3) for RGB image.")

    plt.imshow(rgb_array)
    plt.axis('off')  # Optional: Turn off axis labels
    plt.savefig("./test", bbox_inches='tight', pad_inches=0)
    plt.show()

eval_frequency: int = 10,
unroll_length: int = 5,
batch_size: int = 1024,
num_minibatches: int = 32,
num_update_epochs: int = 4,
reward_scaling: float = 0.1,
entropy_cost: float = 1e-2,
discounting: float = 0.97,
learning_rate: float = 3e-4,
device = "cuda"
env_name = "humanoids"
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']
num_envs = 2
episode_length = 1000

env = envs.create(
        env_name, batch_size=num_envs, episode_length=episode_length, backend="generalized"
    )
env = VectorGymWrapper(env)
env = torch_wrapper.TorchWrapper(env, device=device)
policy_layers = [env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2,]
value_layers = [env.observation_space.shape[-1], 64, 64, 1]
agents = [Agent(policy_layers, value_layers, entropy_cost, discounting, reward_scaling, device).to(device), 
              Agent(policy_layers, value_layers, entropy_cost, discounting, reward_scaling, device).to(device)]
    
agents = [agent.to(device) for agent in agents]

jit_env_reset = jax.jit(env.reset)
jit_env_reset = env.reset
jit_env_step = jax.jit(env.step)

rollout = []
sim_state = jit_env_reset()
breakpoint()
img = save_rgb_image(env.render())