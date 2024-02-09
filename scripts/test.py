import torch

# Humandoid MARL
from Humanoid_MARL import envs
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
from Humanoid_MARL.utils.visual import save_video, save_rgb_image
from Humanoid_MARL.utils.torch_utils import save_models, load_models
from Humanoid_MARL.agent.ppo.train_torch import Agent, eval_unroll, get_agent_actions
from Humanoid_MARL.envs.torch_wrapper import TorchWrapper
from IPython.display import HTML, clear_output
from brax.io import html
import jax
from Humanoid_MARL import envs



config = {
        'num_timesteps': 100_000_000,
        'eval_frequency': 10,
        'episode_length': 1000,
        'unroll_length': 10,
        'num_minibatches': 32,
        'num_update_epochs': 8,
        'discounting': 0.97,
        'learning_rate': 3e-4,
        'entropy_cost': 1e-3,
        'num_envs': 2048,
        'batch_size': 512,
        'env_name': "humanoid",
        'render' : True,
        'device' : 'cuda',
        'model_path' : "./models/20240209_125150_ppo_humanoid.pt",
        'video_length' : 300,
    }
env = envs.create(
        config['env_name'], batch_size=None, episode_length=None, backend="generalized", auto_reset=False,
    )
env = GymWrapper(env, get_jax_state=True)
env = TorchWrapper(env, device=config['device'], get_jax_state=True)

# env warmup
observation = env.reset()
action = torch.zeros(env.action_space.shape[0] * env.num_agents).to(config['device'])
env.step(action)
agents = load_models(config['model_path'], Agent, device=config['device'])
jax_states = []
num_steps = 1000

eval_reward = 0.0
for i in range(num_steps):
    print(f"{i} / {num_steps}")
    logits, action = get_agent_actions(agents, observation, env.obs_dims)
    jax_state, observation, reward, done, info = env.step(Agent.dist_postprocess(action[0]))
    jax_states.append(jax_state)
    print(f"{i} | {info} | DONE [{done}] | Reward [{reward}]")
    eval_reward += reward
print(f"Total Reward | {eval_reward}")
