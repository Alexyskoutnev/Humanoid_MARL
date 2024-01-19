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

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple, Union, List

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
from brax.v1 import envs as envs_v1
import jax
import jax.numpy as jnp
import numpy as np

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]

def get_obs(obs, dims, num_agents):
    total_obs = sum(dims)
    start_idx = 0
    chunks = []

    for dim in dims:
        chunk_size = dim * num_agents
        chunk = jnp.reshape(obs[:, start_idx: start_idx + chunk_size], (num_agents, -1, dim))
        chunks.append(chunk)
        start_idx += chunk_size

    return jnp.concatenate(chunks, axis=2)


def get_action(obs, key: PRNGKey,
               policies : List[Policy], 
               training_states : List,
               dims = Tuple,
               num_agents : int = 1):
    env_obs = get_obs(obs, dims[:-1], num_agents)
    actions = []
    for policy, obs, train_state in zip(policies, env_obs, training_states):
       breakpoint()
       action = policy(obs, key)
       actions.append(action)
    breakpoint()
  

def actor_step(
    env: Env,
    env_state: State,
    training_states: List,
    policy: List[Policy],
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect data."""
  actions, policy_extras = get_action(env_state.obs, key, policy, training_states, env.dims, env.num_humaniods)
#   actions, policy_extras = policy(env_state.obs, key)

  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras
      })


def generate_unroll(
    env: Env,
    env_state: State,
    policy: List[Policy],
    # params: list[PolicyParams],
    training_states : List,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""
  
  # breakpoint()
  params = [(training_state.normalizer_params, training_state.params.policy) for training_state in training_states]
  eval_policy = [policy(_params) for policy, _params in zip(policy, params)]
  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, training_states, eval_policy, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
  return final_state, data



# TODO: Consider moving this to its own file.
class MultiEvaluator:
  """Class to run evaluations."""

  def __init__(self, eval_env: envs.Env,
               eval_policy_fn: List[Policy], num_eval_envs: int,
               episode_length: int, action_repeat: int, key: PRNGKey):
    """Init.

    Args:
      eval_env: Batched environment to run evals on.
      eval_policy_fn: Function returning the policy from the policy parameters.
      num_eval_envs: Each env will run 1 episode in parallel for each eval.
      episode_length: Maximum length of an episode.
      action_repeat: Number of physics steps per env step.
      key: RNG key.
    """
    self._key = key
    self._eval_walltime = 0.

    eval_env = envs.training.EvalWrapper(eval_env)

    def generate_eval_unroll(
          # policy_params: List[PolicyParams],
                             training_state: List,
                             key: PRNGKey) -> State:
      reset_keys = jax.random.split(key, num_eval_envs)
      eval_first_state = eval_env.reset(reset_keys)
      return generate_unroll(
          eval_env,
          eval_first_state,
          eval_policy_fn,
          training_state,
          key,
          unroll_length=episode_length // action_repeat)[0]
    
    self._generate_eval_unroll = jax.jit(generate_eval_unroll)
    # self._generate_eval_unroll = generate_eval_unroll
    self._steps_per_unroll = episode_length * num_eval_envs

  def run_evaluation(self,
                    #  policy_params: List[PolicyParams],
                     trainingState: List,
                     training_metrics: Metrics,
                     aggregate_episodes: bool = True) -> Metrics:
    """Run one epoch of evaluation."""
    self._key, unroll_key = jax.random.split(self._key)

    t = time.time()
    eval_state = self._generate_eval_unroll(trainingState, unroll_key)
    eval_metrics = eval_state.info['eval_metrics']
    eval_metrics.active_episodes.block_until_ready()
    epoch_eval_time = time.time() - t
    metrics = {}
    for fn in [np.mean, np.std]:
      suffix = '_std' if fn == np.std else ''
      metrics.update(
          {
              f'eval/episode_{name}{suffix}': (
                  fn(value) if aggregate_episodes else value
              )
              for name, value in eval_metrics.episode_metrics.items()
          }
      )
    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
    metrics['eval/epoch_eval_time'] = epoch_eval_time
    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
    self._eval_walltime = self._eval_walltime + epoch_eval_time
    metrics = {
        'eval/walltime': self._eval_walltime,
        **training_metrics,
        **metrics
    }

    return metrics  # pytype: disable=bad-return-type  # jax-ndarray
