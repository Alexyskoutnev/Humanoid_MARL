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

# pylint:disable=g-multiple-import
"""Trains a humanoid to run in the +x direction."""

import os

from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.io import html
from etils import epath
import jax
import functools as ft
from jax import numpy as jp
import mujoco
import base64

from Humanoid_MARL import PACKAGE_ROOT


from brax.envs.wrappers import torch as torch_wrapper
from Humanoid_MARL import envs
from Humanoid_MARL.envs.base_env import GymWrapper, VectorGymWrapper
import cProfile
import torch
import numpy as np

#Debugging Flags
from jax import config
# config.update("jax_debug_nans", True) #Throw NaN if they happen
# config.update("jax_disable_jit", True) #Disable JIT [remove this if you want speed]


class Ants(PipelineEnv):
    def __init__(
        self,
        forward_reward_weight=2.25,
        ctrl_cost_weight=0.0,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.5),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        visual="brax",
        num_humanoids=2,
        **kwargs,
    ):
        humanoid_2_path = os.path.join(PACKAGE_ROOT, "assets", "ants.xml")
        sys = mjcf.load(humanoid_2_path)

        with open(humanoid_2_path, "r") as f_path:
            xml_string = f_path.read()
            mj_model = mujoco.MjModel.from_xml_string(xml_string)

        if visual == "mujoco":
            self.mj_model = mj_model
            self.mj_data = mujoco.MjData(mj_model)
            self.renderer = mujoco.Renderer(mj_model)

        n_frames = 5
        self.num_humaniods = num_humanoids
        self._dims = None
        if exclude_current_positions_from_observation:
            self._position_dim = 20
        else:
            self._position_dim = 24
        self._velocity_dim = 23
        self._com_inertia_dim = 110
        self._com_velocity_dim = 66
        self._q_actuator_dim = 23

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.0015)
            n_frames = 10
            gear = jp.array(
                [
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    350.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            )  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == "mjx":
            sys._model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            sys._model.opt.disableflags = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
            sys._model.opt.iterations = 1
            sys._model.opt.ls_iterations = 4

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jax.Array = jax.random.PRNGKey(seed=1)) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        done, _ = jp.zeros(2)
        zero_init = jp.zeros(2)
        reward = jp.zeros(self.num_humaniods)
        metrics = {
            "forward_reward": zero_init,
            "reward_linvel": zero_init,
            "reward_quadctrl": zero_init,
            "reward_alive": zero_init,
            "x_position": zero_init,
            "y_position": zero_init,
            "distance_from_origin": zero_init,
            "x_velocity": zero_init,
            "y_velocity": zero_init,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def _check_is_healthy(self, pipeline_state, min_z, max_z):
        is_healthy_1 = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy_1 = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy_1)
        is_healthy_2 = jp.where(pipeline_state.x.pos[11, 2] < min_z, 0.0, 1.0)
        is_healthy_2 = jp.where(pipeline_state.x.pos[11, 2] > max_z, 0.0, is_healthy_2)
        is_healthy_1_test = is_healthy_1.astype(jp.int32)
        is_healthy_2_test = is_healthy_2.astype(jp.int32)
        return (is_healthy_1_test & is_healthy_2_test).astype(jp.float32)

    def _control_reward(self, action):
        action = reshape_vector(action, (self.num_humaniods, action.shape[0] // self.num_humaniods),)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action), axis=1)
        return ctrl_cost

    def done_signal(self, is_healthy):
        val = len(is_healthy) - sum(is_healthy)
        if jp.logical_and(0, val):
            return 1.0
        else:
            return 0.0
        
    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[:, 0]

        min_z, max_z = self._healthy_z_range
        is_healthy = self._check_is_healthy(pipeline_state, min_z, max_z)

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward * jp.ones(self.num_humaniods)
        else:
            healthy_reward = self._healthy_reward * is_healthy
            
        ctrl_cost = self._control_reward(action)
        
        obs = self._get_obs(pipeline_state, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[:, 0],
            y_position=com_after[:, 1],
            distance_from_origin=jp.linalg.norm(com_after, axis=1),
            x_velocity=velocity[:, 0],
            y_velocity=velocity[:, 1],
        )

        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _flatten(self, x):
        return jp.ravel(x)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            indices_to_remove = np.array([0, 1, 24, 25]) #Removing CoM x-y for humanoid 1 and 2
            position = position[np.logical_not(np.isin(np.arange(len(position)), indices_to_remove))]
    
        com, inertia, mass_sum, x_i = self._com(pipeline_state)

        if self.num_humaniods == 1:
            com, inertia, mass_sum, x_i = self._com(pipeline_state)
            cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        elif self.num_humaniods > 1:
            com = reshape_vector(com, (self.num_humaniods, 1, -1))
            x_i_pos = reshape_vector(x_i.pos, (self.num_humaniods, -1, 3))
            pos_replace = reshape_vector(self._flatten(x_i_pos - com), (-1, 3))
            cinr = x_i.replace(pos=pos_replace).vmap().do(inertia)
            mass_sum = self._flatten(mass_sum[0])  # double check that mass_sum arent different btw the two robots

        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )

        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )
        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                velocity,
                com_inertia.ravel(),
                com_velocity.ravel(),
                qfrc_actuator,
            ]
        )

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ["spring", "positional"]:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        if (self.num_humaniods) == 1:
            mass_sum = jp.sum(inertia.mass)
            x_i = pipeline_state.x.vmap().do(inertia.transform)
            com = (
                jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
            )
        else:
            inertia_mass = reshape_vector(inertia.mass, (self.num_humaniods, -1, 1))
            mass_sum = jp.sum(inertia_mass, axis=1)
            x_i = pipeline_state.x.vmap().do(inertia.transform)
            x_i_pos = reshape_vector(x_i.pos, (self.num_humaniods, -1, 3))
            com = jp.sum(jax.vmap(jp.multiply)(inertia_mass, x_i_pos), axis=1) / reshape_vector(mass_sum, (-1, 1))
        return com, inertia, mass_sum, x_i

    @property
    def dims(self):
        action_dim = int(self.sys.act_size() / self.num_humaniods)
        return (
            self._position_dim,
            self._velocity_dim,
            self._com_inertia_dim,
            self._com_velocity_dim,
            self._q_actuator_dim,
            action_dim,
        )

    @property
    def obs_dims(self):
        return self.dims[:-1]

    @property
    def action_dim(self):
        return self.dims[-1]

    @dims.setter
    def dims(self, new_dims):
        # Setter method allows setting a new value for dims
        self._dims = new_dims

    @property
    def action_space(self):
        return 17 * self.num_humaniods

@ft.partial(jax.jit, static_argnums=1) 
def reshape_vector(vector, target_shape):
    return jp.reshape(vector, target_shape)

if __name__ == "__main__":
    #===============Config===============
    device = "cuda"
    num_envs = 2048
    episode_length = 1000
    #===============Config===============
    env_name = "humanoids"
    env = envs.create(
        env_name,
        batch_size=num_envs,
        episode_length=episode_length,
        backend="generalized",
    )
    env = VectorGymWrapper(env)
    env = torch_wrapper.TorchWrapper(env, device=device)
    obs = env.reset()

    action = torch.ones((env.action_space.shape[0], env.action_space.shape[1] * 2)).to(device) * 2
    env.step(action)