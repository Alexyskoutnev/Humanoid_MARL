# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper around a Brax GymWrapper, that converts outputs to PyTorch tensors.

This conversion happens directly on-device, without moving values to the CPU.
"""
from typing import Optional

# NOTE: The following line will emit a warning and raise ImportError if `torch`
# isn't available.
from brax.io import torch as brax_torch
import gym
import numpy as np
import torch


class TorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(
        self,
        env: gym.Env,
        device: str,
        get_jax_state: bool = True,
    ):
        """Creates a gym Env to one that outputs PyTorch tensors."""
        super().__init__(env)
        self.device = device
        self.get_jax_state = get_jax_state

    def reset(self):
        obs = super().reset()
        return brax_torch.jax_to_torch(obs, device=self.device)

    def step(self, action):
        action = brax_torch.torch_to_jax(action)
        if self.get_jax_state:
            jax_state, obs, reward, done, info = super().step(action)
        else:
            obs, reward, done, info = super().step(action)
        obs = brax_torch.jax_to_torch(obs, device=self.device)
        reward = brax_torch.jax_to_torch(reward, device=self.device)
        done = brax_torch.jax_to_torch(done, device=self.device)
        info = brax_torch.jax_to_torch(info, device=self.device)
        if not self.get_jax_state:
            return obs, reward, done, info
        elif self.get_jax_state:
            return jax_state, obs, reward, done, info


class HistoryBuffer(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        num_envs: int = 2048,
        obs_size: int = 554,
        history_len: int = 4,
        device: Optional[str] = None,
    ):
        super().__init__(env)
        self.env = env
        self.num_envs = num_envs
        self.history_len = history_len
        self.history_buffer = torch.zeros(
            (history_len, num_envs, obs_size), device=device
        )

    def reset(self):
        obs = self.env.reset()
        self.history_buffer[-1, :, :] = obs
        return self.history_buffer

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.history_buffer = torch.cat(
            (self.history_buffer[1:, :, :], obs.unsqueeze(0)), dim=0
        )
        done_idx = torch.where(done)
        for idx in done_idx:
            self.history_buffer[:, idx, :] = 0
        return self.history_buffer, reward, done, info
