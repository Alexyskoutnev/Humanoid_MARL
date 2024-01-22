import gym.envs.mujoco.ant_v4 as ant_v4
import custom_envs.ant_v4 as ant_v42

ant_v4.AntEnv = ant_v42.AntEnv

from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os

from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG

from utils.utils import make_transition, Dict, RunningMeanStd

os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default='Ant-v4', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default='ppo', help='algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=True, help="(default: False)")
parser.add_argument('--epochs', type=int, default=2000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default=5, help='save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default=1, help='print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default=0.1, help='reward scaling(default : 0.1)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')
agent_args = Dict(parser, args.algo)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
else:
    writer = None

env_chaser = gym.make(args.env_name)
env_runner = gym.make(args.env_name)
action_dim = env_chaser.action_space.shape[0]
state_dim = env_chaser.observation_space.shape[0] + 2   # xy coordinates
state_rms_chaser = RunningMeanStd(state_dim)
state_rms_runner = RunningMeanStd(state_dim)
agent_chaser = agent_runner = None
if args.algo == 'ppo':
    agent_chaser = PPO(writer, device, state_dim, action_dim, agent_args)
    agent_runner = PPO(writer, device, state_dim, action_dim, agent_args)
# elif args.algo == 'sac' :
#     agent = SAC(writer, device, state_dim, action_dim, agent_args)
# elif args.algo == 'ddpg' :
#     from utils.noise import OUNoise
#     noise = OUNoise(action_dim,0)
#     agent = DDPG(writer, device, state_dim, action_dim, agent_args, noise)

import pickle

# IN CASE WANTED TO RETRAIN

# load_epoch = "135"
# agent_runner.load_state_dict(torch.load("./model_weights/agent_runner_" + load_epoch))
# agent_chaser.load_state_dict(torch.load("./model_weights/agent_chaser_" + load_epoch))

if (torch.cuda.is_available()) and (args.use_cuda):
    print("Using CuDA")
    agent_chaser = agent_chaser.cuda()
    agent_runner = agent_runner.cuda()

# state_rms_runner = pickle.load(open("./model_weights/rms_runner_" + load_epoch, 'rb'))
# state_rms_chaser = pickle.load(open("./model_weights/rms_chaser_" + load_epoch, 'rb'))
# if args.load != 'no':
#     agent.load_state_dict(torch.load("./model_weights/"+args.load))


score_lst_chaser = []
score_lst_runner = []
state_lst_chaser = []
state_lst_runner = []


chase_after = 100    # START CHASE REWARD AFTER
chase_reward = False

import pickle
from random import randint
ri = lambda : float(randint(-9, 9))
dt = env_chaser.dt * 0  # remove multiplication to enable velocity updates.
train_chaser = True
train_runner = True
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

def get_health_penalty(state_):
    x, y = state_[:2]
    z = state_[2]
    z_penalty = 100*(z-0.6)**6
    xy_penalty = 0.25 * ((x/10)**2 + (y/10)**2)
    return np.clip(z_penalty + xy_penalty, 0, 0.5)

if agent_args.on_policy == True:
    score_chaser = score_runner = 0.0
    state_chaser_ = (env_chaser.reset())[0]
    # random xy init
    a, b, c, d = get_init_coords()
    env_chaser.set_xy_pos(a, b)         # Custom function (see custom_envs)
    state_chaser_[0], state_chaser_[1] = env_chaser.xy_pos  # Custom function (see custom_envs)

    state_runner_ = (env_runner.reset())[0]
    env_runner.set_xy_pos(c, d)
    state_runner_[0], state_runner_[1] = env_runner.xy_pos

    # Chaser knows runner's coords, runner knows chaser's (relative coords)
    state_chaser_ = augment_state(state_chaser_, state_runner_)
    state_runner_ = augment_state(state_runner_, state_chaser_)
    # Scale the states (works well with NNs)
    state_chaser = np.clip((state_chaser_ - state_rms_chaser.mean) / (state_rms_chaser.var ** 0.5 + 1e-8), -5, 5)
    state_runner = np.clip((state_runner_ - state_rms_runner.mean) / (state_rms_runner.var ** 0.5 + 1e-8), -5, 5)
    dg_chaser = dg_runner = 0.
    for n_epi in range(args.epochs):
        if n_epi > chase_after: chase_reward = True
        reset_count = chaser_reset = runner_reset = closeness_reset = 0
        dg_chaser = dg_runner = 0.

        for t in range(agent_args.traj_length):
            if args.render:
                # env_chaser.render()
                # env_runner.render()
                pass
            state_lst_chaser.append(state_chaser_)
            state_lst_runner.append(state_runner_)
            # chaser
            if train_chaser:
                mu, sigma = agent_chaser.get_action(torch.from_numpy(state_chaser).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action_chaser = dist.sample()
                log_prob_chaser = dist.log_prob(action_chaser).sum(-1, keepdim=True)
                next_state_chaser_, reward_chaser, done_chaser, truncated_chaser, info_chaser = \
                    env_chaser.step(action_chaser.cpu().numpy())
            else:
                next_state_chaser_ = state_chaser_[:-2]
                truncated_chaser = False
                done_chaser = False
                reward_chaser = 1
                action_chaser = torch.randn(env_chaser.action_space.shape)
                log_prob_chaser = log_prob_runner = torch.tensor([100.])
            # how much towards the opponent it moved.
            d_1 = abs(state_chaser_[0] - state_runner_[0]) + \
                  abs(state_chaser_[1] - state_runner_[1])
            d_2 = abs(next_state_chaser_[0] - state_runner_[0]) + \
                  abs(next_state_chaser_[1] - state_runner_[1])
            chaser_reward = 10 * (d_1 - d_2)
            if chase_reward and not done_chaser:
                # switch rewards to penalize being unhealthy
                reward_chaser += np.clip(chaser_reward, 0, 1) - get_health_penalty(next_state_chaser_)
            score_chaser += reward_chaser

            # runner
            if train_runner:
                mu, sigma = agent_runner.get_action(torch.from_numpy(state_runner).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action_runner = dist.sample()
                log_prob_runner = dist.log_prob(action_runner).sum(-1, keepdim=True)
                next_state_runner_, reward_runner, done_runner, truncated_runner, info_runner = \
                    env_runner.step(action_runner.cpu().numpy())
            else:
                next_state_runner_ = state_runner_[:-2]
                done_runner = False
                truncated_runner = False
                reward_runner = 1
                action_runner = torch.randn(env_runner.action_space.shape)
                log_prob_runner = torch.tensor([100.])

            d_1 = abs(state_runner_[0] - state_chaser_[0]) + \
                  abs(state_runner_[1] - state_chaser_[1])
            d_2 = abs(next_state_runner_[0] - state_chaser_[0]) + \
                  abs(next_state_runner_[1] - state_chaser_[1])
            runner_reward = 10 * (d_2 - d_1)
            if chase_reward and not done_runner:
                reward_runner += np.clip(runner_reward, 0, 1) - get_health_penalty(next_state_runner_)

            next_state_runner_ = augment_state(next_state_runner_, next_state_chaser_)
            next_state_runner = np.clip((next_state_runner_ - state_rms_runner.mean) / (state_rms_runner.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state_runner, \
                                         action_runner.cpu().numpy(), \
                                         np.array([reward_runner * args.reward_scaling]), \
                                         next_state_runner, \
                                         np.array([done_runner ]), \
                                         log_prob_runner.detach().cpu().numpy() \
                                         )
            agent_runner.put_data(transition)
            score_runner += reward_runner
            next_state_chaser_ = augment_state(next_state_chaser_,
                                               next_state_runner_)
            next_state_chaser = np.clip(
                (next_state_chaser_ - state_rms_chaser.mean) / (state_rms_chaser.var ** 0.5 + 1e-8), -5, 5)
            transition = make_transition(state_chaser, \
                                         action_chaser.cpu().numpy(), \
                                         np.array([reward_chaser * args.reward_scaling]), \
                                         next_state_chaser, \
                                         np.array([done_chaser ]), \
                                         log_prob_chaser.detach().cpu().numpy() \
                                         )
            agent_chaser.put_data(transition)

            closeness = (abs(next_state_runner_[0] - next_state_chaser_[0]) + \
                            abs(next_state_runner_[1] - next_state_chaser_[1]))

            if done_chaser or done_runner or truncated_chaser or truncated_runner or \
                                                                closeness < 0.5:
                if done_chaser:
                    chaser_reset += 1
                if done_runner:
                    runner_reset += 1
                if closeness < 0.5:
                    closeness_reset += 1
                dg_chaser += abs(a - next_state_chaser_[0]) + abs(b - next_state_chaser_[1])
                dg_runner += abs(c - next_state_runner_[0]) + abs(d - next_state_runner_[1])

                state_chaser_ = (env_chaser.reset())[0]
                # random xy init
                a, b, c, d = get_init_coords()
                env_chaser.set_xy_pos(a, b)
                state_chaser_[0], state_chaser_[1] = env_chaser.xy_pos

                state_runner_ = (env_runner.reset())[0]
                env_runner.set_xy_pos(c, d)
                state_runner_[0], state_runner_[1] = env_runner.xy_pos

                state_chaser_ = augment_state(state_chaser_, state_runner_)
                state_runner_ = augment_state(state_runner_, state_chaser_)
                state_chaser = np.clip((state_chaser_ - state_rms_chaser.mean) / (state_rms_chaser.var ** 0.5 + 1e-8),
                                       -5, 5)
                state_runner = np.clip((state_runner_ - state_rms_runner.mean) / (state_rms_runner.var ** 0.5 + 1e-8),
                                       -5, 5)

                score_lst_chaser.append(score_chaser)
                score_lst_runner.append(score_runner)
                # if args.tensorboard:
                #     writer.add_scalar("score/score", score, n_epi)
                score_chaser = score_runner = 0.
            else:
                state_chaser = next_state_chaser
                state_chaser_ = next_state_chaser_
                state_runner = next_state_runner
                state_runner_ = next_state_runner_

        # agent_chaser.train_net(n_epi)
        agent_runner.train_net(n_epi)
        # if n_epi < 20:            # disable scaler update after a fixed iteration
        state_rms_chaser.update(np.vstack(state_lst_chaser))
        state_rms_runner.update(np.vstack(state_lst_runner))

        if n_epi % args.print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score chaser: {:.1f}".format(n_epi, sum(score_lst_chaser) / (len(score_lst_chaser) or 1)))
            print("# of episode :{}, avg score runner: {:.1f}".format(n_epi, sum(score_lst_runner) / (len(score_lst_runner) or 1)))
            print("  Reset Count", runner_reset+chaser_reset+closeness_reset, " (R, C, L) = ", (runner_reset, chaser_reset, closeness_reset))
            print("  Mean distance difference (R, C)", dg_runner/len(score_lst_chaser), dg_chaser/len(score_lst_chaser))
            print()
            score_lst_chaser = []
            score_lst_runner = []
        if n_epi % args.save_interval == 0 and n_epi != 0:
            torch.save(agent_chaser.state_dict(), './model_weights/agent_chaser_' + str(n_epi))
            torch.save(agent_runner.state_dict(), './model_weights/agent_runner_' + str(n_epi))
            pickle.dump(state_rms_runner, open("./model_weights/rms_runner_"+str(n_epi), 'wb'))
            pickle.dump(state_rms_chaser, open("./model_weights/rms_chaser_"+str(n_epi), 'wb'))
# else : # off policy
#     for n_epi in range(args.epochs):
#         score = 0.0
#         state = (env.reset())[0]
#         done = False
#         while not done:
#             if args.render:
#                 env.render()
#             action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
#             action = action.cpu().detach().numpy()
#             next_state, reward, done, info = env.step(action)
#             transition = make_transition(state,\
#                                          action,\
#                                          np.array([reward*args.reward_scaling]),\
#                                          next_state,\
#                                          np.array([done])\
#                                         )
#             agent.put_data(transition)
#
#             state = next_state
#
#             score += reward
#             if agent.data.data_idx > agent_args.learn_start_size:
#                 agent.train_net(agent_args.batch_size, n_epi)
#         score_lst.append(score)
#         if args.tensorboard:
#             writer.add_scalar("score/score", score, n_epi)
#         if n_epi%args.print_interval==0 and n_epi!=0:
#             print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
#             score_lst = []
#         if n_epi%args.save_interval==0 and n_epi!=0:
#             torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))
