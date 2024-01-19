from datetime import datetime

from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.io import html
from etils import epath
import jax
from jax import numpy as jp
import mujoco
import base64

#Visuals
import mediapy as media
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display
import webbrowser

from Humanoid_MARL.agent.ppo.train import train
from Humanoid_MARL.envs.test_env import Humanoid
import Humanoid_MARL.agent.ppo.network as network

def progress(num_steps, metrics):
    times.append(datetime.now())
    xdata.append(num_steps)
    ydata.append(metrics['eval/episode_reward'])
    clear_output(wait=True)
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.ylim([min_y, max_y])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.plot(xdata, ydata)
    plt.show()


def main():
    #================ Config ================
    env_name = 'humanoid'
    num_robots = 2
    backend_visual = "brax"
    xml_path = f"./Humanoid_MARL/assets/humanoid_{num_robots}.xml"
    #================ Config ================
    env = Humanoid(xml_path=xml_path, num_humanoids=num_robots)
    #================ Env Config ================ 
    jit_env_reset = env.reset
    jit_env_step = env.step
    # ctrl = -0.1 * jp.ones(len(env.sys.init_q) - (7 * num_robots))
    #================ Training Config ==================
    num_timesteps = 50_000_000
    num_evals = 10
    reward_scale = 0.1
    episode_length = 1000
    normalize_observations = True
    action_repeat = 1
    unroll_length = 10
    num_minibatches = 32
    num_updates_per_batch = 8
    discounting = 0.97
    learning_rate = 3e-4
    entropy_cost = 1e-3
    num_envs = 2
    batch_size = 1024
    seed = 1
    #================ Training ==================
    max_y = {'ant': 8000, 'halfcheetah': 8000, 'hopper': 2500, 'humanoid': 13000, 'humanoidstandup': 75_000, 'reacher': 5, 'walker2d': 5000, 'pusher': 0}[env_name]
    min_y = {'reacher': -100, 'pusher': -150}.get(env_name, 0)
    xdata, ydata = [], []
    times = [datetime.now()]
    make_inference_fn, params, _ = train(environment=env, num_timesteps=num_timesteps,
                                         num_evals=num_envs, reward_scaling=reward_scale,
                                         episode_length=episode_length, normalize_observations=normalize_observations,
                                         action_repeat=action_repeat, unroll_length=unroll_length,
                                         num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
                                         discounting=discounting, learning_rate=learning_rate,
                                         entropy_cost=entropy_cost, num_envs=num_envs, batch_size=batch_size,
                                         seed=seed, progress_fn=progress, num_agents=num_robots)

if __name__ == "__main__":
    main()



