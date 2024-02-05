import gym.envs.mujoco.ant_v4 as ant_v4
import custom_envs.ant_v4 as ant_v42
from utils.utils import make_transition, Dict, RunningMeanStd

ant_v4.AntEnv = ant_v42.AntEnv

from configparser import ConfigParser
from argparse import ArgumentParser
from random import randint
ri = lambda : float(randint(-9, 9))
import torch
import gym
import numpy as np
import os

from agents.ppo import PPO
DETERMINISTIC = True

env_chaser = gym.make('Ant-v4', render_mode='human')
env_runner = gym.make('Ant-v4')
action_dim = env_chaser.action_space.shape[0]
state_dim = env_chaser.observation_space.shape[0] + 2   # xy coordinates
state_rms_chaser = RunningMeanStd(state_dim)
state_rms_runner = RunningMeanStd(state_dim)
agent_chaser = agent_runner = None
device = "cpu"
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, 'ppo')
agent_chaser = PPO(None, device, state_dim, action_dim, agent_args)
agent_runner = PPO(None, device, state_dim, action_dim, agent_args)

import pickle

# IN CASE WANTED TO RETRAIN

load_epoch = "9370"
agent_runner.load_state_dict(torch.load("./model_weights/agent_runner_" + load_epoch))
agent_chaser.load_state_dict(torch.load("./model_weights/agent_chaser_" + load_epoch))
state_rms_runner = pickle.load(open("./model_weights/rms_runner_" + load_epoch, 'rb'))
state_rms_chaser = pickle.load(open("./model_weights/rms_chaser_" + load_epoch, 'rb'))

def augment_state(s1, s2):
    # Augment s1 with relative position of s2
    extra = [s2[0] - s1[0], s2[1] - s1[1]]
    return np.concatenate([s1, extra])

def get_init_coords():
    # Randomize initial starting points
    while True:
        a, b, c, d = ri(), ri(), ri(), ri()
        if abs(a-c) + abs(b-d) < 4: continue
        return a, b, c, d
state_chaser_ = (env_chaser.reset())[0]
# random xy init
a, b, c, d = get_init_coords()
env_chaser.set_xy_pos(a, b)  # Custom function (see custom_envs)
state_chaser_[0], state_chaser_[1] = env_chaser.xy_pos  # Custom function (see custom_envs)

state_runner_ = (env_runner.reset())[0]
env_runner.set_xy_pos(c, d)
state_runner_[0], state_runner_[1] = env_runner.xy_pos

# Chaser knows runner's coords, runner knows chaser's (relative coords)
state_chaser_ = augment_state(state_chaser_, state_runner_)
state_runner_ = augment_state(state_runner_, state_chaser_)

import time
for i in range(5000):
    # time.sleep(0.2)
    # Scale the states (works well with NNs)
    state_chaser = np.clip((state_chaser_ - state_rms_chaser.mean) / (state_rms_chaser.var ** 0.5 + 1e-8), -5, 5)
    state_runner = np.clip((state_runner_ - state_rms_runner.mean) / (state_rms_runner.var ** 0.5 + 1e-8), -5, 5)

    mu_chaser, sigma_chaser = agent_chaser.get_action(torch.from_numpy(state_chaser).float().to(device))
    mu_runner, sigma_runner = agent_runner.get_action(torch.from_numpy(state_runner).float().to(device))
    dist_chaser = torch.distributions.Normal(mu_chaser, sigma_chaser[0])
    dist_runner = torch.distributions.Normal(mu_runner, sigma_runner[0])
    action_chaser, action_runner = dist_chaser.sample().cpu().numpy(), dist_runner.sample().cpu().numpy()

    next_state_chaser_, reward_chaser, done_chaser, truncated_chaser, info_chaser = \
        env_chaser.step(action_chaser)
    next_state_runner_, reward_runner, done_runner, truncated_runner, info_runner = \
        env_runner.step(action_runner)
    print (state_runner_[0], state_runner_[1])
    print (next_state_chaser_[0], next_state_chaser_[1])
    print ()

    state_chaser_ = augment_state(next_state_chaser_, next_state_runner_)
    # state_runner_ = augment_state(next_state_runner_, next_state_chaser_)
    env_chaser.render()

    # if done_chaser:
    #     print("Stopped!")
    #     break

