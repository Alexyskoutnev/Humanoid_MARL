import torch

# Humandoid MARL
from Humanoid_MARL import envs
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
from Humanoid_MARL.utils.visual import save_video, save_rgb_image
from Humanoid_MARL.utils.torch_utils import save_models, load_models
from Humanoid_MARL.agent.ppo.train_torch import Agent, eval_unroll
from Humanoid_MARL.envs.torch_wrapper import TorchWrapper

def main(config):
    env = envs.create(
        config['env_name'], batch_size=None, episode_length=config['episode_length'], backend="generalized"
    )
    env = GymWrapper(env, get_jax_state=True)
    env = TorchWrapper(env, device=config['device'])

    # env warmup
    obs = env.reset()
    action = torch.zeros(env.action_space.shape[0] * env.num_agents).to(config['device'])
    env.step(action)
    agents = load_models(config['model_path'], Agent, device=config['device'])
    _, eval_rewards = eval_unroll(agents, env, 
                                  video_length=config['video_length'],
                                render=config['render'], 
                                device=config['device'],
                                get_jax_state=True)
    print(f"eval reward: {eval_rewards:.2f}")

if __name__ == "__main__":
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
        'env_name': "humanoids",
        'render' : True,
        'device' : 'cuda',
        'model_path' : "./models/20240202_130311_ppo.pt",
        'video_length' : 300,
    }
    main(config)