# Copyright 2024 The Brax Authors.
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
"""Trains an ant to run in the +x direction."""
from Humanoid_MARL import PACKAGE_ROOT
import os
import numpy as np

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import jit, lax
from jax import numpy as jp
import mujoco
import functools as ft

# from jax import config
# config.update("jax_disable_jit", True)


class Linked_Balls(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        forward_reward_weight=1.0,
        chase_reward_weight=1.0,
        tag_reward_weight=0.0,
        chase_reward_inverse=True,
        full_state_other_agents=False,
        reward_scaling=1.0,
        backend="positional",
        **kwargs,
    ):

        linked_ball_path = os.path.join(PACKAGE_ROOT, "assets", "linked_balls_2.xml")
        # linked_ball_path_walls = os.path.join(PACKAGE_ROOT, "assets", "linked_balls_2_walls.xml")
        sys = mjcf.load(linked_ball_path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 10

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)
        self._q_dim = 6
        self._q_vel_dim = 6
        self.num_agents = 2

        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._forward_reward_weight = forward_reward_weight
        self._chase_reward_weight = chase_reward_weight
        self._chase_reward_inverse = chase_reward_inverse
        self._tag_reward_weight = tag_reward_weight
        self._dims = None
        self._full_state_other_agents = full_state_other_agents

        if full_state_other_agents:
            self._q_dim = 6
            self._q_vel_dim = 6
            self._q_other = 6

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        _, done, zero = jp.zeros(3)
        dummy_val = jp.zeros(2)
        reward = jp.zeros(2)
        metrics = {
            "reward_forward": dummy_val,
            "x_position": dummy_val,
            "y_position": dummy_val,
            "distance_from_origin": dummy_val,
            "x_velocity": dummy_val,
            "y_velocity": dummy_val,
            "reward_chase": dummy_val,
            "reward_tag": dummy_val,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def _tag_reward(self, pipeline_state, threshold=2.0):
        norm = jp.linalg.norm(self._norm(pipeline_state))
        is_below_threshold = jp.any(norm < threshold)
        threshold_int = is_below_threshold.astype(jp.float32)
        return threshold_int * jp.array([1.0, -1.0]) * self._tag_reward_weight

    def _get_forward_reward(
        self, pipeline_state: base.State, pipeline_state0: base.State
    ):
        raise NotImplementedError("Forward reward not implemented.")

    def _chase_reward_fn(self, pipeline_state):
        _dist_diff = jp.sqrt(
            (pipeline_state.x.pos[0, 0] - pipeline_state.x.pos[2, 0]) ** 2
            + (pipeline_state.x.pos[0, 1] - pipeline_state.x.pos[2, 1]) ** 2
        )
        if self._chase_reward_inverse:
            persuader_reward = jp.exp(-_dist_diff * 0.1) * self._chase_reward_weight
            evader_reward = -(jp.exp(-_dist_diff * 0.1) * self._chase_reward_weight)
        else:
            persuader_reward = -_dist_diff * self._chase_reward_weight
            evader_reward = _dist_diff * self._chase_reward_weight
        return jp.concatenate([persuader_reward.reshape(-1), evader_reward.reshape(-1)])

    def _get_velocity_x(self, pipeline_state: base.State, pipeline_state0: base.State):
        delta_x_a_1 = (
            pipeline_state.x.pos[0][0] - pipeline_state0.x.pos[0][0]
        ) / self.dt
        delta_x_a_2 = (
            pipeline_state.x.pos[2][0] - pipeline_state0.x.pos[2][0]
        ) / self.dt
        return jp.concatenate([delta_x_a_1.reshape(-1), delta_x_a_2.reshape(-1)])

    def _get_velocity_y(self, pipeline_state: base.State, pipeline_state0: base.State):
        delta_y_a_1 = pipeline_state.x.pos[0][1] - pipeline_state0.x.pos[0][1]
        delta_y_a_2 = pipeline_state.x.pos[2][1] - pipeline_state0.x.pos[2][1]
        return jp.concatenate([delta_y_a_1.reshape(-1), delta_y_a_2.reshape(-1)])

    def _control_reward(self, action):
        raise NotImplementedError("Control reward not implemented.")

    def _norm(self, pipeline_state: base.State):
        com_a_1 = jp.concatenate(
            [
                pipeline_state.x.pos[0, 0].reshape(-1),
                pipeline_state.x.pos[0, 1].reshape(-1),
            ]
        )
        com_a_2 = jp.concatenate(
            [
                pipeline_state.x.pos[2, 0].reshape(-1),
                pipeline_state.x.pos[2, 1].reshape(-1),
            ]
        )
        norm_origin_distance_a_1 = jp.linalg.norm(com_a_1).reshape(-1)
        norm_origin_distance_a_2 = jp.linalg.norm(com_a_2).reshape(-1)
        norm = jp.concatenate([norm_origin_distance_a_1, norm_origin_distance_a_2])
        return norm

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # # velocity = self._get_forward_reward(pipeline_state, pipeline_state0)
        velocity = self._get_velocity_x(pipeline_state, pipeline_state0)
        velocity_x = self._get_velocity_x(pipeline_state, pipeline_state0)
        velocity_y = self._get_velocity_y(pipeline_state, pipeline_state0)
        forward_reward = velocity * self._forward_reward_weight
        tag_reward = self._tag_reward(pipeline_state)

        # ctrl_cost = self._control_reward(action)
        chase_reward = self._chase_reward_fn(pipeline_state)
        obs = self._get_obs(pipeline_state)
        # reward = forward_reward + healthy_reward - ctrl_cost + tag_reward + chase_reward
        reward = forward_reward + chase_reward + tag_reward
        # done = 1.0 - env_done if self._terminate_when_unhealthy else 0.0
        done = 0.0

        x_pos = jp.concatenate(
            [
                pipeline_state.x.pos[0, 0].reshape(-1),
                pipeline_state.x.pos[2, 0].reshape(-1),
            ]
        )
        y_pos = jp.concatenate(
            [
                pipeline_state.x.pos[0, 1].reshape(-1),
                pipeline_state.x.pos[2, 1].reshape(-1),
            ]
        )

        norm = self._norm(pipeline_state)

        state.metrics.update(
            reward_forward=forward_reward,
            x_position=x_pos,
            y_position=y_pos,
            distance_from_origin=norm,
            x_velocity=velocity_x,
            y_velocity=velocity_y,
            reward_chase=chase_reward,
            reward_tag=tag_reward,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe point mass body position and velocities."""
        qpos = pipeline_state.x.pos.ravel()
        qvel = pipeline_state.xd.vel.ravel()
        if self._full_state_other_agents:
            a1_pos = qpos[0 : qpos.shape[0] // 2]
            a2_pos = qpos[qpos.shape[0] // 2 :]
            positions_other_agents = jp.concatenate([a2_pos, a1_pos])
            return jp.concatenate([qpos] + [qvel] + [positions_other_agents])
        return jp.concatenate([qpos] + [qvel])

    @property
    def dims(self):
        action_dim = int(self.sys.act_size() // self.num_agents)
        if self._full_state_other_agents:
            return (
                self._q_dim,
                self._q_vel_dim,
                self._q_other,
                action_dim,
            )
        else:
            return (action_dim,)

    @property
    def obs_dims(self):
        return self.dims[:-1]

    @property
    def action_dim(self):
        return self.dims[-1]


@ft.partial(jax.jit, static_argnums=1)
def reshape_vector(vector, target_shape):
    return jp.reshape(vector, target_shape)
