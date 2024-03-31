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


class Point_Mass(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
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

        # ant_path = os.path.join(PACKAGE_ROOT, "assets", "ants_2.xml")
        ant_path_wall = os.path.join(PACKAGE_ROOT, "assets", "point_mass.xml")
        sys = mjcf.load(ant_path_wall)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 10

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._q_dim = 3
        self.num_agents = 1

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._forward_reward_weight = forward_reward_weight
        self._chase_reward_weight = chase_reward_weight
        self._chase_reward_inverse = chase_reward_inverse
        self._tag_reward_weight = tag_reward_weight
        self.num_agents = 2
        self._dims = None
        self._or_done_flag = False
        self._and_done_flag = True
        self._full_state_other_agents = full_state_other_agents

        if exclude_current_positions_from_observation:
            self._q_dim = 13
            self._q_vel_dim = 14
            self._q_other = 15

        # if self._use_contact_forces:
        #     raise NotImplementedError("use_contact_forces not implemented.")

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
        dummy_val = jp.zeros(1)
        reward = jp.zeros(1)
        # zero_init = jp.zeros(2)
        metrics = {
            "reward_forward": dummy_val,
            "reward_survive": dummy_val,
            "reward_ctrl": dummy_val,
            "x_position": dummy_val,
            "y_position": dummy_val,
            "distance_from_origin": dummy_val,
            "x_velocity": dummy_val,
            "y_velocity": dummy_val,
            "forward_reward": dummy_val,
            "reward_chase": dummy_val,
            "reward_tag": dummy_val,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def _tag_reward(self, pipeline_state, threshold=2.0):
        raise NotImplementedError("Tag reward not implemented.")

    def _get_forward_reward(
        self, pipeline_state: base.State, pipeline_state0: base.State
    ):
        raise NotImplementedError("Forward reward not implemented.")

    def _chase_reward_fn(self, pipeline_state):
        raise NotImplementedError("Chase reward not implemented.")

    def _get_velocity_x(self, pipeline_state: base.State, pipeline_state0: base.State):
        raise NotImplementedError("Velocity x not implemented.")

    def _get_velocity_y(self, pipeline_state: base.State, pipeline_state0: base.State):
        raise NotImplementedError("Velocity y not implemented.")

    def _control_reward(self, action):
        raise NotImplementedError("Control reward not implemented.")

    def _check_is_healthy(self, pipeline_state, min_z, max_z):
        raise NotImplementedError("Check is healthy not implemented.")

    def _norm(self, pipeline_state: base.State):
        raise NotImplementedError("Norm not implemented.")

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # # velocity = self._get_forward_reward(pipeline_state, pipeline_state0)
        # velocity = self._get_velocity_x(pipeline_state, pipeline_state0)
        # velocity_x = self._get_velocity_x(pipeline_state, pipeline_state0)
        # velocity_y = self._get_velocity_y(pipeline_state, pipeline_state0)
        # forward_reward = velocity * self._forward_reward_weight
        # min_z, max_z = self._healthy_z_range
        # is_healthy, env_done = self._check_is_healthy(pipeline_state, min_z, max_z)
        # if self._terminate_when_unhealthy:
        #     healthy_reward = (
        #         self._healthy_reward * jp.ones(self.num_agents) * is_healthy
        #     )
        # else:
        #     healthy_reward = (
        #         self._healthy_reward * jp.ones(self.num_agents) * is_healthy
        #     )

        # tag_reward = self._tag_reward(pipeline_state)

        # ctrl_cost = self._control_reward(action)
        # contact_cost = 0.0  # TODO: Implement contact cost
        # chase_reward = self._chase_reward_fn(pipeline_state)
        dummy_val = jp.zeros(1)
        obs = self._get_obs(pipeline_state)
        # reward = forward_reward + healthy_reward - ctrl_cost + tag_reward + chase_reward
        reward = jp.zeros(1)
        # done = 1.0 - env_done if self._terminate_when_unhealthy else 0.0
        done = 0.0

        # x_pos = jp.concatenate(
        #     [
        #         pipeline_state.x.pos[0, 0].reshape(-1),
        #         pipeline_state.x.pos[9, 0].reshape(-1),
        #     ]
        # )
        # y_pos = jp.concatenate(
        #     [
        #         pipeline_state.x.pos[0, 1].reshape(-1),
        #         pipeline_state.x.pos[9, 1].reshape(-1),
        #     ]
        # )

        # norm = self._norm(pipeline_state)

        state.metrics.update(
            reward_forward=dummy_val,
            reward_survive=dummy_val,
            reward_ctrl=-dummy_val,
            x_position=dummy_val,
            y_position=dummy_val,
            distance_from_origin=dummy_val,
            x_velocity=dummy_val,
            y_velocity=dummy_val,
            reward_chase=dummy_val,
            reward_tag=dummy_val,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe point mass body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd
        # if self._exclude_current_positions_from_observation:
        #     indices_to_remove = np.array(
        #         [0, 1, 15, 16]
        #     )  # Removing CoM x-y for ant 1 and 2
        #     qpos = qpos[
        #         np.logical_not(np.isin(np.arange(len(qpos)), indices_to_remove))
        #     ]
        # if self._full_state_other_agents:
        #     positions = pipeline_state.q
        #     a1_pos = positions[0:15]
        #     a2_pos = positions[15:30]
        #     positions_other_agents = jp.concatenate([a2_pos, a1_pos])
        #     return jp.concatenate([qpos] + [qvel] + [positions_other_agents])  # noqa
        return jp.concatenate([qpos] + [qvel])

    @property
    def dims(self):
        action_dim = int(self.sys.act_size() // self.num_agents)
        if self._full_state_other_agents:
            return (
                self._q_dim,
                # self._q_vel_dim,
                # self._q_other,
                action_dim,
            )
        else:
            return (
                # self._q_dim,
                # self._q_vel_dim,
                action_dim,
            )

    @property
    def obs_dims(self):
        return self.dims[:-1]

    @property
    def action_dim(self):
        return self.dims[-1]


@ft.partial(jax.jit, static_argnums=1)
def reshape_vector(vector, target_shape):
    return jp.reshape(vector, target_shape)
