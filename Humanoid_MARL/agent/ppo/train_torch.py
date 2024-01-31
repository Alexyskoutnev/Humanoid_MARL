import collections
from datetime import datetime
import functools
import math
import os
import time
import wandb
from typing import Any, Callable, Dict, Optional, Sequence, Union, List, Tuple

import brax

from brax.envs.wrappers import torch as torch_wrapper
from brax.io import metrics
from brax.training.agents.ppo import train as ppo
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# Humandoid MARL
from Humanoid_MARL import envs
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
from Humanoid_MARL.utils.visual import save_video, save_rgb_image
from Humanoid_MARL.utils.torch_utils import save_models, load_models


StepData = collections.namedtuple(
    "StepData", ("observation", "logits", "action", "reward", "done", "truncation")
)


class Agent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
        device: str,
    ):
        super(Agent, self).__init__()

        policy = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(nn.Linear(w1, w2))
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)

        value = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(nn.Linear(w1, w2))
            value.append(nn.SiLU())
        value.pop()  # drop the final activation
        self.value = nn.Sequential(*value)

        self.num_steps = torch.zeros((), device=device)
        self.running_mean = torch.zeros(policy_layers[0], device=device)
        self.running_variance = torch.zeros(policy_layers[0], device=device)

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.3
        self.device = device

    @torch.jit.export
    def dist_create(self, logits):
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc, scale):
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x):
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc, scale):
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc, scale, dist):
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation):
        self.num_steps += observation.shape[0] * observation.shape[1]
        input_to_old_mean = observation - self.running_mean
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        self.running_variance = self.running_variance + var_diff

    @torch.jit.export
    def normalize(self, observation):
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    @torch.jit.export
    def get_logits_action(self, observation):
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(self, truncation, termination, reward, values, bootstrap_value):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = (
            reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        )
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = (
                deltas[ti]
                + self.discounting
                * (1 - termination[ti])
                * truncation_mask[ti]
                * self.lambda_
                * acc
            )
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (
            reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: Dict[str, torch.Tensor], agent_idx : int):
        observation = self.normalize(td["observation"][:,:,agent_idx,:])
        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td["reward"][:,:,agent_idx] * self.reward_scaling
        termination = td["done"] * (1 - td["truncation"])

        action = td["action"].reshape(td["action"].shape[0], td["action"].shape[1], td["observation"].shape[-2], -1)
        action_agent_idx = action[:,:,agent_idx,:]
        td_logits = td["logits"].reshape(td["logits"].shape[0], td["logits"].shape[1], td["observation"].shape[-2], -1)
        td_logit_agent_idx = td_logits[:,:,agent_idx,:]
        
        loc, scale = self.dist_create(td_logit_agent_idx)
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, action_agent_idx)
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, action_agent_idx)

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td["truncation"],
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy

        return policy_loss + v_loss + entropy_loss

def add_extra_dimension(data, idx : int = 1):
    if len(data.shape) == 2:
        return data.unsqueeze(idx)
    elif len(data.shape) == 3:
        return data.unsqueeze(idx + 1)

def sd_map(f: Callable[..., torch.Tensor], *sds, add_dim: bool = False) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    if add_dim:
        for k in keys:
            items[k] = add_extra_dimension(items[k])
    return StepData(**items)

def sd_map_minibatch(f: Callable[..., torch.Tensor], *sds, **kwargs) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    idx_into = lambda d: d[kwargs['minibatch_idx']]
    for k in keys:
        if k == 'observation':
            field_data = [sd._asdict()[k] for sd in sds]
            items[k] = f(*field_data, **kwargs)
        else:
            field_data = [sd._asdict()[k] for sd in sds]
            items[k] = idx_into(*field_data)
    return StepData(**items)

def eval_unroll(agents : List[Agent],
                env : Union[VectorGymWrapper, GymWrapper],
                length : int = 1000,
                device : str = 'cpu',
                render : bool = False,
                video_length : int = 100) -> Union[torch.Tensor, float]:
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    episodes = torch.zeros((), device=device)
    episode_reward = torch.zeros((), device=device)
    frames = []
    for i in range(length):
        logits, action = get_agent_actions(agents, observation, env.obs_dims)
        observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
        if render and i < video_length:
            print(f"Img cnt : {i}")
            img = env.render() #We have to figure why the this is so slow (slows down the RL-Pipeline)
            frames.append(img)
    try:
        save_video(frames)
    except:
        print("Failed to save video")
    return episodes, episode_reward / episodes

def get_obs(obs, dims, num_agents):
    total_obs = sum(dims)
    start_idx = 0
    chunks = []
    for dim in dims:
        if len(obs.shape) == 1:
            chunk_size = dim * num_agents
            chunk = torch.reshape(obs[start_idx: start_idx + chunk_size], (num_agents, dim))
            chunks.append(chunk)
            start_idx += chunk_size
        elif len(obs.shape) > 1:
            chunk_size = dim * num_agents
            chunk = torch.reshape(obs[:, start_idx: start_idx + chunk_size], (num_agents, -1, dim))
            chunks.append(chunk)
            start_idx += chunk_size
    if len(obs.shape) == 1:
        return torch.concatenate(chunks, axis=1) #Assuming only 1 enviroment [obs]
    elif len(obs.shape) >= 1:
        return torch.concatenate(chunks, axis=-1).transpose(0, 1) #Parallized Enviroments [#envs, #num_agents, obs]

def get_agent_actions(agents : List[Agent], observation : torch.Tensor, dims : Tuple[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    num_agents = len(agents)
    obs = get_obs(observation, dims, num_agents) # [#num_agents, obs]
    logits, actions = [], []
    for idx, agent in enumerate(agents):
        logit, action = agent.get_logits_action(obs[:,idx,:])
        logits.append(logit)
        actions.append(action)
    return torch.concatenate(logits, axis=1), torch.concatenate(actions, axis=1)

def train_unroll(agents, env, observation, num_unrolls, unroll_length, add_dim=False):
    """Return step data over multple unrolls."""
    sd = StepData([], [], [], [], [], [])
    for _ in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for _ in range(unroll_length):
            logits, action = get_agent_actions(agents, observation, env.obs_dims)
            observation, reward, done, info = env.step(Agent.dist_postprocess(action))
            one_unroll.observation.append(observation)
            one_unroll.logits.append(logits)
            one_unroll.action.append(action)
            one_unroll.reward.append(reward)
            one_unroll.done.append(done)
            one_unroll.truncation.append(info["truncation"])
        # Apply torch.stack to each field in one_unroll
        one_unroll = sd_map(torch.stack, one_unroll)
        # Update the overall StepData structure by concatenating data from the current unroll
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
    # Apply torch.stack to each field in sd
    td = sd_map(torch.stack, sd, add_dim=add_dim)
    return observation, td

def update_normalization(agents : List[Agent], observation : torch.Tensor, dims : Tuple[int]):
    num_agents = len(agents)
    observation = observation.view(observation.shape[0] *  observation.shape[1], -1)
    obs = get_obs(observation, dims, num_agents)
    for idx, agent in enumerate(agents):
        agent.update_normalization(obs[:,idx,:])

def unroll_first(data):
    data = data.swapaxes(0, 1)
    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))

def reshape_minibatch(epoch_td : torch.Tensor, minibatch_idx : int, dims : Tuple[int] = (24, 23, 110, 66, 23), num_agents : int = 2) -> torch.Tensor:
    """
    Reshape and index into a minibatch of trajectories.

    Args:
        epoch_td (torch.Tensor): Tensor representing a minibatch of trajectories with shape [minibatch_dim, unroll_length_dim, batch_size, num_agents, agent_obs_dim].
        minibatch_idx (int): Index of the desired minibatch.
        dims (Tuple[int], optional): Dimensions to reshape the observation tensor. Defaults to (24, 23, 110, 66, 23).
        num_agents (int, optional): Number of agents in the trajectories. Defaults to 2.

    Returns:
        torch.Tensor: Reshaped and indexed minibatch of trajectories.
    """
    minibatch_dim, unroll_length_dim, batch_size =  epoch_td.shape[0], epoch_td.shape[1], epoch_td.shape[2] 
    observation = epoch_td.reshape(-1, epoch_td.shape[3])
    epoch_td = get_obs(observation, dims, num_agents)
    epoch_td = epoch_td.reshape(*(minibatch_dim,unroll_length_dim,batch_size), num_agents, -1)
    return epoch_td[minibatch_idx,:,:,:,:]

def train(
    env_name: str = "humanoids",
    num_envs: Union[int, None] = 2048,
    episode_length: int = 1000,
    device: str = "cuda",
    num_timesteps: int = 30_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = 0.1,
    entropy_cost: float = 1e-2,
    discounting: float = 0.97,
    learning_rate: float = 3e-4,
    render : bool = False,
    debug : bool = False,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Trains a policy via PPO."""
    env = envs.create(
        env_name, batch_size=num_envs, episode_length=episode_length, backend="generalized"
    )
    
    env = VectorGymWrapper(env)
    # automatically convert between jax ndarrays and torch tensors:
    env = torch_wrapper.TorchWrapper(env, device=device)

    # env warmup
    obs = env.reset()
    action = torch.zeros((env.action_space.shape[0], env.action_space.shape[1] * env.num_agents)).to(device)
    env.step(action)
    
    # create the agent
    policy_layers = [env.observation_space.shape[-1], 64, 64, env.action_space.shape[-1] * 2,]
    value_layers = [env.observation_space.shape[-1], 64, 64, 1]
    network_arch = {"policy_layers": policy_layers,
                    "value_layers": value_layers,
                    "entropy_cost": entropy_cost,
                    "discounting": discounting,
                    "reward_scaling": reward_scaling,
                    "device": device}
    agents = [Agent(**network_arch).to(device), Agent(**network_arch).to(device)]
    save_models(agents, network_arch)
    if not debug:
        agents = [torch.jit.script(agent.to(device)) for agent in agents] #Only uncomment once whole pipeline is implemented
    else:
        agents = [agent.to(device) for agent in agents] #Only uncomment once whole pipeline is implemented

    optimizers = [optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents]

    sps = 0
    total_steps = 0
    total_loss = 0
    for eval_i in range(eval_frequency + 1):

        if progress_fn:
            t = time.time()
            with torch.no_grad():
                episode_count, episode_reward = eval_unroll(agents, env, episode_length, device, render=render)
            duration = time.time() - t
            episode_avg_length = env.num_envs * episode_length / episode_count
            eval_sps = env.num_envs * episode_length / duration
            progress = {
                "eval/episode_reward": episode_reward,
                "eval/completed_episodes": episode_count,
                "eval/avg_episode_length": episode_avg_length,
                "speed/sps": sps,
                "speed/eval_sps": eval_sps,
                "losses/total_loss": total_loss,
            }
            if not debug:
                wandb.log({"eval/episode_reward" : episode_reward,
                        "speed/sps": sps,
                        "speed/eval_sps": eval_sps,
                        "losses/total_loss": total_loss})
            progress_fn(total_steps, progress)

        if eval_i == eval_frequency:
            break

        observation = env.reset()        
        num_steps = batch_size * num_minibatches * unroll_length 
        num_epochs = num_timesteps // (num_steps * eval_frequency)
        num_unrolls = batch_size * num_minibatches // env.num_envs
        total_loss = 0
        t = time.time()
        for num_epoch in range(num_epochs):
            observation, td = train_unroll(agents, env, observation, num_unrolls, unroll_length)
            td = sd_map(unroll_first, td)
            # update normalization statistics
            update_normalization(agents, td.observation, env.obs_dims)
            epoch_loss = 0.0
            for update_epoch in range(num_update_epochs):
                # shuffle and batch the data
                with torch.no_grad():
                    permutation = torch.randperm(td.observation.shape[1], device=device)

                    def shuffle_batch(data):
                        data = data[:, permutation]
                        data = data.reshape([data.shape[0], num_minibatches, -1] + list(data.shape[2:]))
                        return data.swapaxes(0, 1)
                    
                    epoch_td = sd_map(shuffle_batch, td)

                for minibatch_i in range(num_minibatches):
                    td_minibatch = sd_map_minibatch(reshape_minibatch, epoch_td, minibatch_idx=minibatch_i, dims=env.obs_dims, num_agents=env.num_agents)
                    for idx, (agent, optimizer) in enumerate(zip(agents, optimizers)):
                        loss = agent.loss(td_minibatch._asdict(), agent_idx=idx)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss
                        epoch_loss += loss
            if not debug:
                wandb.log({"training/epoch-loss": epoch_loss})
            print(f"epoch {num_epoch} : [{epoch_loss}]")


        duration = time.time() - t
        total_steps += num_epochs * num_steps
        total_loss = total_loss / ((num_epochs * num_update_epochs * num_minibatches) + 1)
        sps = num_epochs * num_steps / duration
    save_models(agents, network_arch)


