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

"""Wrappers to convert brax envs to gym envs."""
from typing import ClassVar, Optional

from brax.envs.base import PipelineEnv
from brax.io import image
import gym
from gym import spaces
from gym.vector import utils
import jax
import numpy as np

import jax
from jax import numpy as jp
from jax import vmap
from jax.tree_util import tree_map

def take(input, i, axis=0): #Brax version of .take() doesn't work
    return tree_map(lambda x: jp.take(x, i, axis=axis, mode='wrap'), input)

class GymWrapper(gym.Env):
  """A wrapper that converts Brax Env to one that follows Gym API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: PipelineEnv,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.num_agents = env.num_humaniods
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.dt
    }
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs = np.inf * np.ones(self._env.observation_size // self.num_agents, dtype='float32')
    self.observation_space = spaces.Box(-obs, obs, dtype='float32')
    self.obs_dims = env.obs_dims

    action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
    self.num_actuators = len(self._env.sys.actuator.ctrl_range) // self.num_agents
    self.action_space = spaces.Box(action[:, 0][:self.num_actuators], action[:, 1][:self.num_actuators], dtype='float32')
    self.action_dim = env.action_dim

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    # We return device arrays for pytorch users.
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    # We return device arrays for pytorch users.
    return obs, reward, done, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='rgb_array'):
    if mode == 'rgb_array':
      sys, state = self._env.sys, self._state
      if state is None:
        raise RuntimeError('must call reset or step before rendering')
      return image.render_array(sys, state.pipeline_state, 512, 512)
    else:
      return super().render(mode=mode)  # just raise an exception

class VectorGymWrapper(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(self,
               env: PipelineEnv,
               seed: int = 0,
               backend: Optional[str] = None):
    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.dt
    }
    if not hasattr(self._env, 'batch_size'):
      raise ValueError('underlying env must be batched')

    self.num_envs = self._env.batch_size
    self.seed(seed)
    self.backend = backend
    self._state = None

    self.num_agents = env.num_humaniods

    obs = np.inf * np.ones(self._env.observation_size // self.num_agents, dtype='float32')
    obs_space = spaces.Box(-obs, obs, dtype='float32')
    self.observation_space = utils.batch_space(obs_space, self.num_envs)
    self.obs_dims = env.obs_dims

    action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
    self.num_actuators = len(self._env.sys.actuator.ctrl_range) // self.num_agents
    action_space = spaces.Box(action[:, 0][:self.num_actuators], action[:, 1][:self.num_actuators], dtype='float32')
    # action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')
    self.action_space = utils.batch_space(action_space, self.num_envs)
    self.action_dim = env.action_dim

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(state, action):
      state = self._env.step(state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.reward, state.done, info

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, self._key = self._reset(self._key)
    return obs

  def step(self, action):
    self._state, obs, reward, done, info = self._step(self._state, action)
    return obs, reward, done, info

  def seed(self, seed: int = 0):
    self._key = jax.random.PRNGKey(seed)

  def render(self, mode='rgb_array'):
    if mode == 'rgb_array':
      sys, state = self._env.sys, self._state
      if state is None:
        raise RuntimeError('must call reset or step before rendering')
      return image.render_array(sys, take(state.pipeline_state, 0), 512, 512)
    else:
      return super().render(mode=mode)  # just raise an exception