import torch

# Humandoid MARL
from Humanoid_MARL import envs
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
from Humanoid_MARL.utils.visual import save_video, save_rgb_image
from Humanoid_MARL.utils.torch_utils import save_models, load_models
from Humanoid_MARL.agent.ppo.train_torch import eval_unroll, get_agent_actions
from Humanoid_MARL.agent.ppo.agent import Agent
# from Humanoid_MARL.envs.torch_wrapper import TorchWrapper
from brax.envs.wrappers import torch as torch_wrapper
from IPython.display import HTML, clear_output
from brax.io import html
import jax
from Humanoid_MARL import envs


env_name = "humanoid"

config = {
            'num_timesteps': 150_000_000,
            'eval_reward_limit' : 15_000,
            'eval_frequency': 100,
            'episode_length': 1000,
            'unroll_length': 10,
            'num_minibatches': 32,
            'num_update_epochs': 8,
            'discounting': 0.97,
            'learning_rate': 3e-4,
            'entropy_cost': 2e-3,
            'num_envs': 2048,
            'batch_size': 512,
            'env_name': env_name,
            'device' : 'cuda',
            'device_idx' : 0,
            'model_path' : "./models/20240212_214953_ppo_humanoid.pt",
        }
env = envs.create(
        env_name,
        batch_size=config['num_envs'],
        episode_length=config['episode_length'],
        backend="generalized",
        device_idx=0,
    )
env = VectorGymWrapper(env)
env = torch_wrapper.TorchWrapper(env, device=config['device'])
obs = env.reset()
action = torch.zeros(
        (env.action_space.shape[0], env.action_space.shape[1] * env.num_agents)
    ).to(config['device'])
env.step(action)

# breakpoint()
agents = load_models(config['model_path'], Agent, device=config['device'])
agents = [torch.jit.script(agent.to(config['device'])) for agent in agents]
# agents = [agent.eval() for agent in agents]

with torch.no_grad():
                episode_count, episode_reward = eval_unroll(
                    agents, env, config['episode_length'], config['device']
                )
print(f"Episode {episode_count} reward: {episode_reward}")


