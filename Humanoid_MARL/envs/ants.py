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


class Ants(PipelineEnv):
    """
    ### Description

    This environment is based on the environment introduced by Schulman, Moritz,
    Levine, Jordan and Abbeel in
    ["High-Dimensional Continuous Control Using Generalized Advantage Estimation"](https://arxiv.org/abs/1506.02438).

    The ant is a 3D robot consisting of one torso (free rotational body) with four
    legs attached to it with each leg having two links.

    The goal is to coordinate the four legs to move in the forward (right)
    direction by applying torques on the eight hinges connecting the two links of
    each leg and the torso (nine parts and eight hinges).

    ### Action Space

    The agent take a 8-element vector for actions.

    The action space is a continuous `(action, action, action, action, action,
    action, action, action)` all in `[-1, 1]`, where `action` represents the
    numerical torques applied at the hinge joints.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
    |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the rotor between the torso and front left hip   | -1          | 1           | hip_1 (front_left_leg)           | hinge | torque (N m) |
    | 1   | Torque applied on the rotor between the front left two links       | -1          | 1           | ankle_1 (front_left_leg)         | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the torso and front right hip  | -1          | 1           | hip_2 (front_right_leg)          | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the front right two links      | -1          | 1           | ankle_2 (front_right_leg)        | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between the torso and back left hip    | -1          | 1           | hip_3 (back_leg)                 | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the back left two links        | -1          | 1           | ankle_3 (back_leg)               | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the torso and back right hip   | -1          | 1           | hip_4 (right_back_leg)           | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between the back right two links       | -1          | 1           | ankle_4 (right_back_leg)         | hinge | torque (N m) |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    ant, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(27,)` where the elements correspond to the following:

    | Num | Observation                                                  | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
    |-----|--------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the torso (centre)                           | -Inf | Inf | torso                            | free  | position (m)             |
    | 1   | w-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
    | 2   | x-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
    | 3   | y-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
    | 4   | z-orientation of the torso (centre)                          | -Inf | Inf | torso                            | free  | angle (rad)              |
    | 5   | angle between torso and first link on front left             | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
    | 6   | angle between the two links on the front left                | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
    | 7   | angle between torso and first link on front right            | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
    | 8   | angle between the two links on the front right               | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
    | 9   | angle between torso and first link on back left              | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
    | 10  | angle between the two links on the back left                 | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
    | 11  | angle between torso and first link on back right             | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
    | 12  | angle between the two links on the back right                | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |
    | 13  | x-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
    | 14  | y-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
    | 15  | z-coordinate velocity of the torso                           | -Inf | Inf | torso                            | free  | velocity (m/s)           |
    | 16  | x-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
    | 17  | y-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
    | 18  | z-coordinate angular velocity of the torso                   | -Inf | Inf | torso                            | free  | angular velocity (rad/s) |
    | 19  | angular velocity of angle between torso and front left link  | -Inf | Inf | hip_1 (front_left_leg)           | hinge | angle (rad)              |
    | 20  | angular velocity of the angle between front left links       | -Inf | Inf | ankle_1 (front_left_leg)         | hinge | angle (rad)              |
    | 21  | angular velocity of angle between torso and front right link | -Inf | Inf | hip_2 (front_right_leg)          | hinge | angle (rad)              |
    | 22  | angular velocity of the angle between front right links      | -Inf | Inf | ankle_2 (front_right_leg)        | hinge | angle (rad)              |
    | 23  | angular velocity of angle between torso and back left link   | -Inf | Inf | hip_3 (back_leg)                 | hinge | angle (rad)              |
    | 24  | angular velocity of the angle between back left links        | -Inf | Inf | ankle_3 (back_leg)               | hinge | angle (rad)              |
    | 25  | angular velocity of angle between torso and back right link  | -Inf | Inf | hip_4 (right_back_leg)           | hinge | angle (rad)              |
    | 26  | angular velocity of the angle between back right links       | -Inf | Inf | ankle_4 (right_back_leg)         | hinge | angle (rad)              |

    The (x,y,z) coordinates are translational DOFs while the orientations are
    rotational DOFs expressed as quaternions.

    ### Rewards

    The reward consists of three parts:

    - *reward_survive*: Every timestep that the ant is alive, it gets a reward of
      1.
    - *reward_forward*: A reward of moving forward which is measured as
      *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the
      time between actions - the default *dt = 0.05*. This reward would be
      positive if the ant moves forward (right) desired.
    - *reward_ctrl*: A negative reward for penalising the ant if it takes actions
      that are too large. It is measured as *coefficient **x**
      sum(action<sup>2</sup>)* where *coefficient* is a parameter set for the
      control and has a default value of 0.5.
    - *contact_cost*: A negative reward for penalising the ant if the external
      contact force is too large. It is calculated *0.5 * 0.001 *
      sum(clip(external contact force to [-1,1])<sup>2</sup>)*.

    ### Starting State

    All observations start in state (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a
    uniform noise in the range of [-0.1, 0.1] added to the positional values and
    standard normal noise with 0 mean and 0.1 standard deviation added to the
    velocity values for stochasticity.

    Note that the initial z coordinate is intentionally selected to be slightly
    high, thereby indicating a standing up ant. The initial orientation is
    designed to make it face forward as well.

    ### Episode Termination

    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. The y-orientation (index 2) in the state is **not** in the range
       `[0.2, 1.0]`
    """

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
        backend="positional",
        **kwargs,
    ):

        # ant_path = os.path.join(PACKAGE_ROOT, "assets", "ants_2.xml")
        ant_path_wall = os.path.join(PACKAGE_ROOT, "assets", "ants_2_walls.xml")
        sys = mjcf.load(ant_path_wall)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 10

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

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

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

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
        reward = jp.zeros(2)
        zero_init = jp.zeros(2)
        metrics = {
            "reward_forward": zero_init,
            "reward_survive": zero_init,
            "reward_ctrl": zero_init,
            "x_position": zero_init,
            "y_position": zero_init,
            "distance_from_origin": zero_init,
            "x_velocity": zero_init,
            "y_velocity": zero_init,
            "forward_reward": zero_init,
            "reward_chase": zero_init,
            "reward_tag": zero_init,
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
        delta_x = pipeline_state.x.pos[0][0] - pipeline_state0.x.pos[0][0]
        delta_y = pipeline_state.x.pos[0][1] - pipeline_state0.x.pos[0][1]
        agent_0_v_norm = jp.sqrt(
            (delta_x / (self.dt + 0.001)) ** 2 + (delta_y / (self.dt + 0.001)) ** 2
        )
        delta_x = pipeline_state.x.pos[9][0] - pipeline_state0.x.pos[9][0]
        delta_y = pipeline_state.x.pos[9][1] - pipeline_state0.x.pos[9][1]
        agent_1_v_norm = jp.sqrt(
            (delta_x / (self.dt + 0.001)) ** 2 + (delta_y / (self.dt + 0.001)) ** 2
        )
        return jp.concatenate([agent_0_v_norm.reshape(-1), agent_1_v_norm.reshape(-1)])

    def _chase_reward_fn(self, pipeline_state):
        _dist_diff = jp.sqrt(
            (jp.abs(pipeline_state.x.pos[0, 0]) - jp.abs(pipeline_state.x.pos[9, 0]))
            ** 2
            + (jp.abs(pipeline_state.x.pos[0, 1]) - jp.abs(pipeline_state.x.pos[9, 1]))
            ** 2
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
            pipeline_state.x.pos[9][0] - pipeline_state0.x.pos[9][0]
        ) / self.dt
        return jp.concatenate([delta_x_a_1.reshape(-1), delta_x_a_2.reshape(-1)])

    def _get_velocity_y(self, pipeline_state: base.State, pipeline_state0: base.State):
        delta_y_a_1 = pipeline_state.x.pos[0][1] - pipeline_state0.x.pos[0][1]
        delta_y_a_2 = pipeline_state.x.pos[9][1] - pipeline_state0.x.pos[9][1]
        return jp.concatenate([delta_y_a_1.reshape(-1), delta_y_a_2.reshape(-1)])

    def _control_reward(self, action):
        action = reshape_vector(
            action,
            (self.num_agents, action.shape[0] // self.num_agents),
        )
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action), axis=1)
        return ctrl_cost

    def _check_is_healthy(self, pipeline_state, min_z, max_z):
        is_healthy_1 = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy_1 = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy_1)
        is_healthy_2 = jp.where(pipeline_state.x.pos[9, 2] < min_z, 0.0, 1.0)
        is_healthy_2 = jp.where(pipeline_state.x.pos[9, 2] > max_z, 0.0, is_healthy_2)
        is_healthy_1_test = is_healthy_1.astype(jp.int32)
        is_healthy_2_test = is_healthy_2.astype(jp.int32)
        done_signals = jp.concatenate(
            [is_healthy_1_test.reshape(-1), is_healthy_2_test.reshape(-1)]
        )
        if self._or_done_flag:
            env_done = (is_healthy_1_test | is_healthy_2_test).astype(jp.float32)
            return done_signals, env_done
        elif self._and_done_flag:
            env_done = (is_healthy_1_test & is_healthy_2_test).astype(jp.float32)
            return done_signals, env_done
        else:
            env_done = (is_healthy_1_test & is_healthy_2_test).astype(jp.float32)
            return done_signals, env_done

    def _norm(self, pipeline_state: base.State):
        com_a_1 = jp.concatenate(
            [
                pipeline_state.x.pos[0, 0].reshape(-1),
                pipeline_state.x.pos[0, 1].reshape(-1),
            ]
        )
        com_a_2 = jp.concatenate(
            [
                pipeline_state.x.pos[9, 0].reshape(-1),
                pipeline_state.x.pos[9, 1].reshape(-1),
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

        # velocity = self._get_forward_reward(pipeline_state, pipeline_state0)
        velocity = self._get_velocity_x(pipeline_state, pipeline_state0)
        velocity_x = self._get_velocity_x(pipeline_state, pipeline_state0)
        velocity_y = self._get_velocity_y(pipeline_state, pipeline_state0)
        forward_reward = velocity * self._forward_reward_weight
        min_z, max_z = self._healthy_z_range
        is_healthy, env_done = self._check_is_healthy(pipeline_state, min_z, max_z)
        if self._terminate_when_unhealthy:
            healthy_reward = (
                self._healthy_reward * jp.ones(self.num_agents) * is_healthy
            )
        else:
            healthy_reward = (
                self._healthy_reward * jp.ones(self.num_agents) * is_healthy
            )

        tag_reward = self._tag_reward(pipeline_state)

        ctrl_cost = self._control_reward(action)
        contact_cost = 0.0  # TODO: Implement contact cost
        chase_reward = self._chase_reward_fn(pipeline_state)

        obs = self._get_obs(pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost + tag_reward + chase_reward
        done = 1.0 - env_done if self._terminate_when_unhealthy else 0.0

        x_pos = jp.concatenate(
            [
                pipeline_state.x.pos[0, 0].reshape(-1),
                pipeline_state.x.pos[9, 0].reshape(-1),
            ]
        )
        y_pos = jp.concatenate(
            [
                pipeline_state.x.pos[0, 1].reshape(-1),
                pipeline_state.x.pos[9, 1].reshape(-1),
            ]
        )

        norm = self._norm(pipeline_state)

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
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
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd
        if self._exclude_current_positions_from_observation:
            indices_to_remove = np.array(
                [0, 1, 15, 16]
            )  # Removing CoM x-y for ant 1 and 2
            qpos = qpos[
                np.logical_not(np.isin(np.arange(len(qpos)), indices_to_remove))
            ]
        if self._full_state_other_agents:
            positions = pipeline_state.q
            a1_pos = positions[0:15]
            a2_pos = positions[15:30]
            positions_other_agents = jp.concatenate([a2_pos, a1_pos])
            return jp.concatenate([qpos] + [qvel] + [positions_other_agents])  # noqa
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
            return (
                self._q_dim,
                self._q_vel_dim,
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
