import collections
from queue import Queue
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
from brax.envs.base import Env

# Humandoid MARL
from Humanoid_MARL import envs
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
from Humanoid_MARL.utils.torch_utils import save_models, load_models
from Humanoid_MARL.utils.logger import WandbLogger
from Humanoid_MARL.agent.ppo.agent import Agent
from Humanoid_MARL.agent.ppo.agent_lstm import AgentLSTM
from Humanoid_MARL.envs.torch_wrapper import TorchWrapper, HistoryBuffer

StepData = collections.namedtuple(
    "StepData", ("observation", "logits", "action", "reward", "done", "truncation")
)


class SavingModelException(Exception):
    pass


def combine(data):
    if len(data.shape) == 3:  # combing done and truncatio
        return data.flatten()
    else:  # Combining logits, action, reward
        return data.reshape([-1] + [data.shape[-1]])


def combine_alt(data):
    data = data.swapaxes(2, 3)
    return data.reshape([-1] + list(data.shape[-2:]))


def unroll_first(data):
    data = data.swapaxes(0, 1)
    return data.reshape([data.shape[0], -1] + list(data.shape[3:]))


def unroll_first_time_series(data):
    data = data.swapaxes(2, 3)  # put time right before observation space
    data = data.reshape(
        tuple(data.shape[:3]) + (-1,)
    )  # convert from [8, 3, 1024, 4, 554] to [8, 3, 1024, 4 * 554]
    return unroll_first(data)


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)


def sd_map_alt(
    f: Callable[..., torch.Tensor], f_alt: Callable[..., torch.Tensor], *sds
) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        if k == "observation":
            items[k] = f_alt(*[sd._asdict()[k] for sd in sds])
        else:
            items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)


def sd_map_minibatch(f: Callable[..., torch.Tensor], *sds, **kwargs) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    idx_into = lambda d: d[kwargs["minibatch_idx"]]
    for k in keys:
        if k == "observation":
            field_data = [sd._asdict()[k] for sd in sds]
            items[k] = f(*field_data, **kwargs)
        else:
            field_data = [sd._asdict()[k] for sd in sds]
            items[k] = idx_into(*field_data)
    return StepData(**items)


def eval_unroll(
    agents: List[Agent],
    env: Union[VectorGymWrapper, GymWrapper],
    length: int = 1000,
    device: str = "cpu",
    get_jax_state: bool = False,
    get_full_state: bool = False,
    agent_config: Dict[str, Any] = {},
) -> Union[torch.Tensor, float]:
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    episodes = torch.zeros((), device=device)
    episode_reward = torch.zeros((), device=device)
    frames = []
    num_resets = 1
    for i in range(length):
        _, action = get_agent_actions(
            agents, observation, env.obs_dims_tuple, get_full_state, agent_config={}
        )
        if get_jax_state:
            jax_state, observation, reward, done, _ = env.step(
                Agent.dist_postprocess(action)
            )
        else:
            observation, reward, done, _ = env.step(Agent.dist_postprocess(action))
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
    if get_jax_state:
        return episodes, episode_reward / episodes, jax_state
    else:
        return episodes, episode_reward / episodes


def get_obs(
    obs: torch.Tensor, dims: Tuple[int], num_agents: int, get_full_state: bool = False
) -> torch.Tensor:
    if get_full_state:
        return obs
    if type(dims) == int:  # TODO: FIX THIS, just assume one type of observation for now
        total_obs = dims
    else:
        total_obs = sum(dims)
    start_idx = 0
    chunks = []
    for dim in dims:
        chunk_size = dim * num_agents
        chunk = torch.reshape(
            obs[:, start_idx : start_idx + chunk_size], (-1, num_agents, dim)
        ).swapaxes(0, 1)
        chunks.append(chunk)
        start_idx += chunk_size
    if len(obs.shape) == 1:
        return torch.concatenate(chunks, axis=1)  # Assuming only 1 enviroment [obs]
    elif len(obs.shape) >= 1:
        return torch.concatenate(chunks, axis=-1).transpose(
            0, 1
        )  # Parallized Enviroments [#envs, #num_agents, obs]


def _tranform_observation(
    obs: torch.Tensor, dims: Tuple[int], num_agents: int = 2
) -> torch.Tensor:
    start_idx = 0
    chunks = []
    for dim in dims:
        chunk_size = dim * num_agents
        chunk = torch.reshape(
            obs[:, start_idx : start_idx + chunk_size], (-1, num_agents, dim)
        ).swapaxes(0, 1)
        chunks.append(chunk)
        start_idx += chunk_size
    return torch.concatenate(chunks, axis=-1).transpose(
        0, 1
    )  # Parallized Enviroments [#envs, #num_agents, obs]


def get_obs_time_series(
    obs: torch.Tensor, dims: Tuple[int], num_agents: int = 2
) -> torch.Tensor:
    """Convert gym obs space of [seq_length, obs_dim]
    to [seq_length, num_agents, obs_dim] with
    dimensional matching."""
    seq_len = obs.shape[0]
    _history = obs[:-1]
    _observation = obs[-1]
    _observation = _tranform_observation(_observation, dims, num_agents)
    observation = torch.zeros((seq_len,) + _observation.shape, device=obs.device)
    for idx, _obs_history in enumerate(_history):
        if _obs_history.shape != _observation.shape:
            _obs_history = _tranform_observation(_obs_history, dims, num_agents)
        observation[idx] = _obs_history
    return observation.swapaxes(0, 1)


def get_agent_actions(
    agents: List[Union[Agent, AgentLSTM]],
    observation: torch.Tensor,
    dims: Tuple[int],
    get_full_state: bool = False,
    agent_config: Dict[str, Any] = {},
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_agents = len(agents)
    if isinstance(agents[0], Agent) or agents[0].original_name == "Agent":
        if num_agents == 1:
            agent = agents[0]
            if len(observation.shape) == 1:  # Used for single enviroment for evaluation
                observation = observation.reshape(1, -1)
            logit, action = agent.get_logits_action(observation)
            return logit, action
        elif num_agents > 1:
            if len(observation.shape) == 1:  # Used for single enviroment for evaluation
                observation = observation.reshape(1, -1)
            observation = get_obs(
                observation, dims, num_agents, get_full_state=get_full_state
            )  # [#envs, #num_agents, obs]
            logits, actions = [], []
            for idx, agent in enumerate(agents):
                if len(observation.shape) == 2:
                    logit, action = agent.get_logits_action(observation)
                else:
                    logit, action = agent.get_logits_action(observation[:, idx, :])
                if agent_config.get("freeze_idx") == idx:
                    action = torch.ones_like(action) * 0.1
                logits.append(logit)
                actions.append(action)
            return torch.concatenate(logits, axis=1), torch.concatenate(actions, axis=1)
    elif isinstance(agents[0], AgentLSTM) or agents[0].original_name == "AgentLSTM":
        breakpoint()
        observation = get_obs(
            observation, dims, num_agents, get_full_state=get_full_state
        )
        logits, actions = [], []
        for idx, agent in enumerate(agents):
            logit, action = agent.get_logits_action(observation[:, :, idx, :])
            if agent_config.get("freeze_idx") == idx:
                action = torch.ones_like(action) * 0.1
            logits.append(logit)
            actions.append(action)
        return torch.concatenate(logits, axis=1), torch.concatenate(actions, axis=1)
    else:
        raise ValueError(
            f"Agents must be of type Agent or AgentLSTM, {agents[0]} found."
        )


def train_unroll(
    agents,
    env,
    observation,
    num_unrolls,
    unroll_length,
    debug=False,
    logger=None,
    get_full_state=False,
    agent_config={},
):
    """Return step data over multple unrolls."""
    sd = StepData([], [], [], [], [], [])
    for _ in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for i in range(unroll_length):
            logits, action = get_agent_actions(
                agents, observation, env.obs_dims_tuple, get_full_state, agent_config
            )
            observation, reward, done, info = env.step(Agent.dist_postprocess(action))
            one_unroll.observation.append(observation)
            one_unroll.logits.append(logits)
            one_unroll.action.append(action)
            one_unroll.reward.append(reward)
            one_unroll.done.append(done)
            one_unroll.truncation.append(info["truncation"])
            if not debug:
                logger.log_train(info=info, rewards=reward, num_agents=len(agents))
        # Apply torch.stack to each field in one_unroll
        one_unroll = sd_map(torch.stack, one_unroll)
        # Update the overall StepData structure by concatenating data from the current unroll
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
    # Apply torch.stack to each field in sd
    td = sd_map(torch.stack, sd)
    return observation, td


def update_normalization_time_series(
    agents: List[Agent],
    observation: torch.Tensor,
    dims: Tuple[int],
) -> None:
    num_agents = len(agents)
    observation = observation.view(
        (observation.shape[0] * observation.shape[1],) + tuple(observation.shape[2:])
    )
    if num_agents == 1:
        raise NotImplementedError
    else:
        obs = get_obs_time_series(observation, dims, num_agents)
        for idx, agent in enumerate(agents):
            agent.update_normalization(obs[:, :, idx, :])


def update_normalization(
    agents: List[Agent],
    observation: torch.Tensor,
    dims: Tuple[int],
    get_full_state: bool = False,
) -> None:
    num_agents = len(agents)
    observation = observation.view(observation.shape[0] * observation.shape[1], -1)
    if num_agents == 1:
        agent = agents[0]
        agent.update_normalization(observation)
    else:
        obs = get_obs(observation, dims, num_agents, get_full_state=get_full_state)
        for idx, agent in enumerate(agents):
            if get_full_state:
                agent.update_normalization(obs[:])
            else:
                agent.update_normalization(obs[:, idx, :])


def reshape_minibatch(
    epoch_td: torch.Tensor,
    minibatch_idx: int,
    dims: Tuple[int],
    num_agents: int = 2,
    get_full_state: bool = False,
) -> torch.Tensor:
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
    minibatch_dim, unroll_length_dim, batch_size = (
        epoch_td.shape[0],
        epoch_td.shape[1],
        epoch_td.shape[2],
    )
    observation = epoch_td.reshape(
        -1, epoch_td.shape[3]
    )  # from [minibatch_dim, unroll_length_dim, batch_size, agent_obs_dim] -> [minibatch_dim * unroll_length_dim * batch_size, agent_obs_dim]
    if num_agents > 1:  # convert to two agent observations
        epoch_td = get_obs(observation, dims, num_agents, get_full_state=get_full_state)
    epoch_td = epoch_td.reshape(
        *(minibatch_dim, unroll_length_dim, batch_size), num_agents, -1
    )  # [minibatch_dim, unroll_length_dim, batch_dim, num_agent, agent_obs] : [32, 3, 256, 2, 277]
    return epoch_td[
        minibatch_idx, :, :, :, :
    ]  # -> torch.Size([3, 256, 2, 277]) : [unroll_length_dim, batch_dim, num_agent, agent_obs]


def reshape_minibatch_time_series(
    epoch_td: torch.Tensor,
    minibatch_idx: int,
    dims: Tuple[int],
    seq_length=4,
    num_agents: int = 2,
) -> torch.Tensor:
    """
    Reshapes a time-series tensor representing minibatch data into a format suitable for processing
    time-series data for a specified number of agents.

    Args:
        epoch_td (torch.Tensor): Input tensor representing minibatch time-series data.
                                  It has a shape of [minibatch_dim, unroll_length_dim, batch_size, agent_obs_dim].
        minibatch_idx (int): Index of the minibatch to be extracted.
        dims (Tuple[int]): Dimensions of the reshaped tensor.
        seq_length (int, optional): Length of the sequence. Default is 4.
        num_agents (int, optional): Number of agents. Default is 2.

    Returns:
        torch.Tensor: Reshaped tensor representing the time-series data.
    """
    minibatch_dim, unroll_length_dim, batch_size = (
        epoch_td.shape[0],
        epoch_td.shape[1],
        epoch_td.shape[2],
    )
    observation = epoch_td.reshape(
        -1, epoch_td.shape[3]
    )  # from [minibatch_dim, unroll_length_dim, batch_size, agent_obs_dim] -> [minibatch_dim * unroll_length_dim * batch_size, agent_obs_dim]
    batch_dim = observation.shape[0]
    observation = observation.reshape(batch_dim, seq_length, -1)  # [24576, 4, 554]
    epoch_td = get_obs_time_series(
        observation, dims, num_agents
    )  # [seq_length, batch_dim, num_agent, agent_obs] : example -> epoch_td.shape = torch.Size([4, 24576, 2, 277])
    epoch_td = epoch_td.swapaxes(
        0, 1
    )  # put batch dim as zero index : [batch dim, seq_length, num_agent, agent_obs] : example -> epoch_td.shape = torch.Size([24576, 4, 2, 277])
    epoch_td = epoch_td.reshape(
        (minibatch_dim, unroll_length_dim, batch_size) + tuple(epoch_td.shape[-3:])
    )  # [32, 3, 256, 4, 2, 277]
    return epoch_td[
        minibatch_idx, :, :, :
    ]  # [unroll_length, batch_dim, seq_len, num_agent, agent_obs] : example -> epoch_td.shape = torch.size([3, 256, 4, 2, 277])


def setup_env(
    env_name: str,
    num_envs: int,
    episode_length: int,
    device_idx: int,
    env_config: Dict,
    full_state: bool,
    device="cuda",
) -> Env:
    env = envs.create(
        env_name,
        batch_size=num_envs,
        episode_length=episode_length,
        device_idx=device_idx,
        **env_config,
    )
    env = VectorGymWrapper(env, full_state=full_state)
    # automatically convert between jax ndarrays and torch tensors:
    env = torch_wrapper.TorchWrapper(env, device=device)
    return env


def setup_agents(
    env_name: str,
    device: str,
    model_path: str,
    reward_scaling: float,
    entropy_cost: float,
    discounting: float,
    learning_rate: float,
    env: Env,
    debug: bool = False,
    network_config={},
) -> Tuple[List[Agent], List[optim.Optimizer]]:

    network_arch = {
        "policy_layers": network_config["POLICY_LAYERS"],
        "value_layers": network_config["VALUE_LAYERS"],
        "entropy_cost": entropy_cost,
        "discounting": discounting,
        "reward_scaling": reward_scaling,
        "device": device,
        "network_config": network_config,
    }

    def create_agent() -> Union[Agent, AgentLSTM]:
        if network_config.get("LSTM"):
            return AgentLSTM(**network_arch).to(device)
        else:
            return Agent(**network_arch).to(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{timestamp}_ppo_{env_name}.pt"

    optimizers = []
    agents = []

    if model_path:
        try:
            agents = load_models(model_path, Agent, device=device)
            optimizers = [
                optim.Adam(agent.parameters(), lr=float(learning_rate))
                for agent in agents
            ]
        except Exception as e:
            print(f"Failed to load model: {e}")
            exit()
    else:
        agents = [create_agent() for _ in range(env.num_agents)]
        optimizers = [
            optim.Adam(agent.parameters(), lr=learning_rate) for agent in agents
        ]
    if not debug:
        agents = [torch.jit.script(agent.to(device)) for agent in agents]
    return agents, optimizers, model_name, network_arch


def train(
    env_name: str = "humanoids",
    num_envs: Union[int, None] = 2048,
    episode_length: int = 1000,
    device: str = "cuda",
    num_timesteps: int = 100_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 512,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = 0.1,
    entropy_cost: float = 1e-3,
    discounting: float = 0.97,
    learning_rate: float = 3e-4,
    eval_reward_limit: float = 10_000,
    debug: bool = False,
    device_idx: int = 0,
    logger=None,
    notebook: bool = False,
    model_path: str = None,
    env_config: Dict[str, Any] = {},
    eval_flag: bool = True,
    runnning_avg_length: int = 3,
    full_state: bool = False,  # Enable every agent see eac
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    agent_config: Dict[str, Any] = {},
    network_config: Dict[str, Any] = {},
    time_series: bool = False,
) -> List[Agent]:

    debug = False
    """Trains a policy via PPO."""
    env = setup_env(
        env_name,
        num_envs,
        episode_length,
        device_idx,
        env_config,
        full_state,
        device=device,
    )
    # ========= env warmup (for JIT) =========
    obs = env.reset()
    print("Env dims: ", env.obs_dims)
    action = torch.zeros(
        (env.action_space.shape[0], env.action_space.shape[1] * env.num_agents)
    ).to(device)
    env.step(action)
    # ========= env warmup (for JIT) =========
    # ========= training vars ================
    running_mean = list()
    sps = 0
    total_steps = 0
    total_loss = 0
    # ========= training vars ================
    # ========= create the agent =============
    agents, optimizers, model_name, network_arch = setup_agents(
        env_name,
        device,
        model_path,
        reward_scaling,
        entropy_cost,
        discounting,
        learning_rate,
        env,
        debug=debug,
        network_config=network_config,
    )
    # ========= create the agent =============
    for eval_i in range(eval_frequency + 1):

        observation = env.reset()
        num_steps = batch_size * num_minibatches * unroll_length
        num_epochs = num_timesteps // (num_steps * eval_frequency)
        if num_epochs <= 0:
            raise ValueError(
                "num_timesteps too low for given batch size and unroll length"
            )
        num_unrolls = batch_size * num_minibatches // env.num_envs
        total_loss = 0
        t = time.time()
        print("num_unrolls: ", num_unrolls)
        print("unroll_length: ", unroll_length)
        print("num_epochs: ", num_epochs)
        for num_epoch in range(num_epochs):
            observation, td = train_unroll(
                agents,
                env,
                observation,
                num_unrolls,
                unroll_length,
                debug=debug,
                logger=logger,
                get_full_state=full_state,
                agent_config=agent_config,
            )
            epoch_loss = 0.0
            # if time_series:
            #     td = sd_map_alt(unroll_first, unroll_first_time_series, td)
            #     for update_epoch in range(num_update_epochs):
            #         # shuffle and batch the data
            #         with torch.no_grad():
            #             permutation = torch.randperm(
            #                 td.observation.shape[1], device=device
            #             )

            #             def shuffle_batch(data):
            #                 data = data[:, permutation]
            #                 data = data.reshape(
            #                     [data.shape[0], num_minibatches, -1]
            #                     + list(data.shape[2:])
            #                 )
            #                 return data.swapaxes(0, 1)

            #             epoch_td = sd_map(shuffle_batch, td)  # -> [32, 3, 256, 2216]
            #         for minibatch_i in range(num_minibatches):
            #             td_minibatch = sd_map_minibatch(
            #                 reshape_minibatch_time_series,
            #                 epoch_td,
            #                 minibatch_idx=minibatch_i,
            #                 dims=env.obs_dims_tuple,
            #                 num_agents=env.num_agents,
            #             )
            #             for idx, (agent, optimizer) in enumerate(
            #                 zip(agents, optimizers)
            #             ):
            #                 if agent_config.get("freeze_idx") == idx:
            #                     continue
            #                 loss = agent.loss(td_minibatch._asdict(), agent_idx=idx)
            #                 optimizer.zero_grad()
            #                 loss.backward()
            #                 optimizer.step()
            #                 total_loss += loss
            #                 epoch_loss += loss
            # else:
            breakpoint()
            td = sd_map(unroll_first, td)
            # update normalization statistics
            update_normalization(
                agents,
                td.observation,
                env.obs_dims_tuple,
                get_full_state=full_state,
            )
            breakpoint()
            for update_epoch in range(num_update_epochs):
                # shuffle and batch the data
                with torch.no_grad():
                    permutation = torch.randperm(td.observation.shape[1], device=device)

                    def shuffle_batch(data):
                        data = data[:, permutation]
                        data = data.reshape(
                            [data.shape[0], num_minibatches, -1] + list(data.shape[2:])
                        )
                        return data.swapaxes(0, 1)

                    epoch_td = sd_map(shuffle_batch, td)  # -> [32, 3, 256, 554]

                for minibatch_i in range(num_minibatches):
                    td_minibatch = sd_map_minibatch(
                        reshape_minibatch,
                        epoch_td,
                        minibatch_idx=minibatch_i,
                        dims=env.obs_dims_tuple,
                        num_agents=env.num_agents,
                        get_full_state=full_state,
                    )  # -> [3, 256, 2, 277]
                    for idx, (agent, optimizer) in enumerate(zip(agents, optimizers)):
                        if agent_config.get("freeze_idx") == idx:
                            continue
                        loss = agent.loss(td_minibatch._asdict(), agent_idx=idx)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss
                        epoch_loss += loss

            if not debug:
                logger.log_epoch_loss(epoch_loss / num_epoch + 1)
            print(f"epoch {num_epoch} : [{epoch_loss}]")

        duration = time.time() - t
        total_steps += num_epochs * num_steps
        total_loss = total_loss / (
            (num_epochs * num_update_epochs * num_minibatches) + 1
        )
        sps = num_epochs * num_steps / duration
    return agents
