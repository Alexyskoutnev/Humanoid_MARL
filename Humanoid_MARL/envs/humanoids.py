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

# Debugging Flags
from jax import config

# config.update("jax_debug_nans", True) #Throw NaN if they happen
# config.update("jax_disable_jit", True) #Disable JIT [remove this if you want speed]


class Humanoid(PipelineEnv):
    """
    ### Description

    This environment is based on the environment introduced by Tassa, Erez and
    Todorov in
    ["Synthesis and stabilization of complex behaviors through online trajectory optimization"](https://ieeexplore.ieee.org/document/6386025).

    The 3D bipedal robot is designed to simulate a human. It has a torso (abdomen)
    with a pair of legs and arms. The legs each consist of two links, and so the
    arms (representing the knees and elbows respectively). The goal of the
    environment is to walk forward as fast as possible without falling over.

    ### Action Space

    The agent take a 17-element vector for actions. The action space is a
    continuous `(action, ...)` all in `[-1, 1]`, where `action` represents the
    numerical torques applied at the hinge joints.

    | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding config)   | Joint | Unit         |
    |-----|------------------------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the hinge in the y-coordinate of the abdomen                     | -1.0        | 1.0         | abdomen_yz                       | hinge | torque (N m) |
    | 1   | Torque applied on the hinge in the z-coordinate of the abdomen                     | -1.0        | 1.0         | abdomen_yz                       | hinge | torque (N m) |
    | 2   | Torque applied on the hinge in the x-coordinate of the abdomen                     | -1.0        | 1.0         | abdomen_x                        | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between torso/abdomen and the right hip (x-coordinate) | -1.0        | 1.0         | right_hip_xyz (right_thigh)      | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between torso/abdomen and the right hip (y-coordinate) | -1.0        | 1.0         | right_hip_xyz (right_thigh)      | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between torso/abdomen and the right hip (z-coordinate) | -1.0        | 1.0         | right_hip_xyz (right_thigh)      | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the right hip/thigh and the right shin         | -1.0        | 1.0         | right_knee                       | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between torso/abdomen and the left hip (x-coordinate)  | -1.0        | 1.0         | left_hip_xyz (left_thigh)        | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between torso/abdomen and the left hip (y-coordinate)  | -1.0        | 1.0         | left_hip_xyz (left_thigh)        | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between torso/abdomen and the left hip (z-coordinate)  | -1.0        | 1.0         | left_hip_xyz (left_thigh)        | hinge | torque (N m) |
    | 10  | Torque applied on the rotor between the left hip/thigh and the left shin           | -1.0        | 1.0         | left_knee                        | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between the torso and right upper arm (coordinate -1)  | -1.0        | 1.0         | right_shoulder12                 | hinge | torque (N m) |
    | 12  | Torque applied on the rotor between the torso and right upper arm (coordinate -2)  | -1.0        | 1.0         | right_shoulder12                 | hinge | torque (N m) |
    | 13  | Torque applied on the rotor between the right upper arm and right lower arm        | -1.0        | 1.0         | right_elbow                      | hinge | torque (N m) |
    | 14  | Torque applied on the rotor between the torso and left upper arm (coordinate -1)   | -1.0        | 1.0         | left_shoulder12                  | hinge | torque (N m) |
    | 15  | Torque applied on the rotor between the torso and left upper arm (coordinate -2)   | -1.0        | 1.0         | left_shoulder12                  | hinge | torque (N m) |
    | 16  | Torque applied on the rotor between the left upper arm and left lower arm          | -1.0        | 1.0         | left_elbow                       | hinge | torque (N m) |

    ### Observation Space

    The state space consists of positional values of different body parts of the
    Humanoid, followed by the velocities of those individual parts (their
    derivatives) with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(376,)` where the elements correspond to the following:

    | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding config)   | Joint | Unit                     |
    |-----|-----------------------------------------------------------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
    | 0   | z-coordinate of the torso (centre)                                                                              | -Inf | Inf | root                             | free  | position (m)             |
    | 1   | w-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
    | 2   | x-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
    | 3   | y-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
    | 4   | z-orientation of the torso (centre)                                                                             | -Inf | Inf | root                             | free  | angle (rad)              |
    | 5   | z-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_yz                       | hinge | angle (rad)              |
    | 6   | y-angle of the abdomen (in lower_waist)                                                                         | -Inf | Inf | abdomen_yy                       | hinge | angle (rad)              |
    | 7   | x-angle of the abdomen (in pelvis)                                                                              | -Inf | Inf | abdomen_x                        | hinge | angle (rad)              |
    | 8   | x-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_xyz                    | hinge | angle (rad)              |
    | 9   | y-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_xyz                    | hinge | angle (rad)              |
    | 10  | z-coordinate of angle between pelvis and right hip (in right_thigh)                                             | -Inf | Inf | right_hip_xyz                    | hinge | angle (rad)              |
    | 11  | angle between right hip and the right shin (in right_knee)                                                      | -Inf | Inf | right_knee                       | hinge | angle (rad)              |
    | 12  | x-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_xyz                     | hinge | angle (rad)              |
    | 13  | y-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_xyz                     | hinge | angle (rad)              |
    | 14  | z-coordinate of angle between pelvis and left hip (in left_thigh)                                               | -Inf | Inf | left_hip_xyz                     | hinge | angle (rad)              |
    | 15  | angle between left hip and the left shin (in left_knee)                                                         | -Inf | Inf | left_knee                        | hinge | angle (rad)              |
    | 16  | coordinate-1 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder12                 | hinge | angle (rad)              |
    | 17  | coordinate-2 (multi-axis) angle between torso and right arm (in right_upper_arm)                                | -Inf | Inf | right_shoulder12                 | hinge | angle (rad)              |
    | 18  | angle between right upper arm and right_lower_arm                                                               | -Inf | Inf | right_elbow                      | hinge | angle (rad)              |
    | 19  | coordinate-1 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder12                  | hinge | angle (rad)              |
    | 20  | coordinate-2 (multi-axis) angle between torso and left arm (in left_upper_arm)                                  | -Inf | Inf | left_shoulder12                  | hinge | angle (rad)              |
    | 21  | angle between left upper arm and left_lower_arm                                                                 | -Inf | Inf | left_elbow                       | hinge | angle (rad)              |
    | 22  | x-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)           |
    | 23  | y-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)           |
    | 24  | z-coordinate velocity of the torso (centre)                                                                     | -Inf | Inf | root                             | free  | velocity (m/s)           |
    | 25  | x-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s) |
    | 26  | y-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s) |
    | 27  | z-coordinate angular velocity of the torso (centre)                                                             | -Inf | Inf | root                             | free  | angular velocity (rad/s) |
    | 28  | z-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_z                        | hinge | angular velocity (rad/s) |
    | 29  | y-coordinate of angular velocity of the abdomen (in lower_waist)                                                | -Inf | Inf | abdomen_y                        | hinge | angular velocity (rad/s) |
    | 30  | x-coordinate of angular velocity of the abdomen (in pelvis)                                                     | -Inf | Inf | abdomen_x                        | hinge | angular velocity (rad/s) |
    | 31  | x-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_xyz                    | hinge | angular velocity (rad/s) |
    | 32  | y-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_z                      | hinge | angular velocity (rad/s) |
    | 33  | z-coordinate of the angular velocity of the angle between pelvis and right hip (in right_thigh)                 | -Inf | Inf | right_hip_y                      | hinge | angular velocity (rad/s) |
    | 34  | angular velocity of the angle between right hip and the right shin (in right_knee)                              | -Inf | Inf | right_knee                       | hinge | angular velocity (rad/s) |
    | 35  | x-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_xyz                     | hinge | angular velocity (rad/s) |
    | 36  | y-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_z                       | hinge | angular velocity (rad/s) |
    | 37  | z-coordinate of the angular velocity of the angle between pelvis and left hip (in left_thigh)                   | -Inf | Inf | left_hip_y                       | hinge | angular velocity (rad/s) |
    | 38  | angular velocity of the angle between left hip and the left shin (in left_knee)                                 | -Inf | Inf | left_knee                        | hinge | angular velocity (rad/s) |
    | 39  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder12                 | hinge | angular velocity (rad/s) |
    | 40  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and right arm (in right_upper_arm) | -Inf | Inf | right_shoulder12                 | hinge | angular velocity (rad/s) |
    | 41  | angular velocity of the angle between right upper arm and right_lower_arm                                       | -Inf | Inf | right_elbow                      | hinge | angular velocity (rad/s) |
    | 42  | coordinate-1 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder12                  | hinge | angular velocity (rad/s) |
    | 43  | coordinate-2 (multi-axis) of the angular velocity of the angle between torso and left arm (in left_upper_arm)   | -Inf | Inf | left_shoulder12                  | hinge | angular velocity (rad/s) |
    | 44  | angular velocity of the angle between left upper arm and left_lower_arm                                         | -Inf | Inf | left_elbow                       | hinge | angular velocity (rad/s) |

    Additionally, after all the positional and velocity based values in the table,
    the state_space consists of (in order):

    - *cinert:* Mass and inertia of a single rigid body relative to the center of
      mass (this is an intermediate result of transition). It has shape 14*10
      (*nbody * 10*) and hence adds to another 140 elements in the state space.
    - *cvel:* Center of mass based velocity. It has shape 14 * 6 (*nbody * 6*) and
      hence adds another 84 elements in the state space
    - *qfrc_actuator:* Constraint force generated as the actuator force. This has
      shape `(23,)`  *(nv * 1)* and hence adds another 23 elements to the state
      space.

    The (x,y,z) coordinates are translational DOFs while the orientations are
    rotational DOFs expressed as quaternions.

    ### Rewards

    The reward consists of three parts:

    - *reward_alive*: Every timestep that the humanoid is alive, it gets a reward
      of 5.
    - *forward_reward*: A reward of walking forward which is measured as *1.25 *
      (average center of mass before action - average center of mass after
      action) / dt*. *dt* is the time between actions - the default *dt = 0.015*.
      This reward would be positive if the humanoid walks forward (right) desired.
      The calculation for the center of mass is defined in the `.py` file for the
      Humanoid.
    - *reward_quadctrl*: A negative reward for penalising the humanoid if it has
      too large of a control force. If there are *nu* actuators/controls, then the
      control has shape  `nu x 1`. It is measured as *0.1 **x**
      sum(control<sup>2</sup>)*.

    ### Starting State

    All observations start in state (0.0, 0.0,  1.4, 1.0, 0.0  ... 0.0) with a
    uniform noise in the range of [-0.01, 0.01] added to the positional and
    velocity values (values in the table) for stochasticity. Note that the initial
    z coordinate is intentionally selected to be high, thereby indicating a
    standing up humanoid. The initial orientation is designed to make it face
    forward as well.

    ### Episode Termination

    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. The z-coordinate of the torso (index 0 in state space OR index 2 in the
    table) is **not** in the range `[0.8, 2.1]` (the humanoid has fallen or is
    about to fall beyond recovery).
    """

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.5),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        include_standing_up_cost=False,
        backend="generalized",
        visual="brax",
        num_humanoids=2,
        **kwargs,
    ):
        humanoid_2_path = os.path.join(PACKAGE_ROOT, "assets", "humanoid_2_no_collision.xml")
        sys = mjcf.load(humanoid_2_path)

        with open(humanoid_2_path, "r") as f_path:
            xml_string = f_path.read()
            mj_model = mujoco.MjModel.from_xml_string(xml_string)

        if visual == "mujoco":
            self.mj_model = mj_model
            self.mj_data = mujoco.MjData(mj_model)
            self.renderer = mujoco.Renderer(mj_model)

        n_frames = 5
        self.num_humanoids = num_humanoids
        self.num_agents = 2
        self._dims = None
        if exclude_current_positions_from_observation:
            self._position_dim = 22
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
        self._standup_reward = include_standing_up_cost
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
        reward = jp.zeros(self.num_humanoids)
        metrics = {
            "forward_reward": zero_init,
            "reward_linvel": zero_init,
            "reward_quadctrl": zero_init,
            "reward_alive": zero_init,
            "standup_reward": zero_init,
            "x_position": zero_init,
            "y_position": zero_init,
            "distance_from_origin": zero_init,
            "x_velocity": zero_init,
            "y_velocity": zero_init,
            "z_position": zero_init,
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

    def _stand_up_rewards(self, pipeline_state):
        uph_cost_h1 = pipeline_state.x.pos[0, 2] / self.dt
        uph_cost_h2 = pipeline_state.x.pos[11, 2] / self.dt
        return jp.concatenate([uph_cost_h1.reshape(-1), uph_cost_h2.reshape(-1)])

    def _control_reward(self, action):
        action = reshape_vector(
            action,
            (self.num_humanoids, action.shape[0] // self.num_humanoids),
        )
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

        if self._standup_reward:
            uph_cost = self._stand_up_rewards(pipeline_state)
        else:
            uph_cost = jp.zeros(2)

        min_z, max_z = self._healthy_z_range
        is_healthy = self._check_is_healthy(pipeline_state, min_z, max_z)
        if self._terminate_when_unhealthy:
            healthy_reward = (
                self._healthy_reward * jp.ones(self.num_humanoids) * (is_healthy)
            )
        else:
            healthy_reward = (
                self._healthy_reward * jp.ones(self.num_humanoids) * is_healthy
            )

        ctrl_cost = self._control_reward(action)

        obs = self._get_obs(pipeline_state, action)
        reward = forward_reward + healthy_reward - ctrl_cost + uph_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        humanoids_z = jp.concatenate(
            [
                pipeline_state.x.pos[0, 2].reshape(-1),
                pipeline_state.x.pos[11, 2].reshape(-1),
            ]
        )

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
            z_position=humanoids_z,
            standup_reward=uph_cost,
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _flatten(self, x):
        return jp.ravel(x)

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            indices_to_remove = np.array(
                [0, 1, 24, 25]
            )  # Removing CoM x-y for humanoid 1 and 2
            position = position[
                np.logical_not(np.isin(np.arange(len(position)), indices_to_remove))
            ]

        com, inertia, mass_sum, x_i = self._com(pipeline_state)

        if self.num_humanoids == 1:
            com, inertia, mass_sum, x_i = self._com(pipeline_state)
            cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        elif self.num_humanoids > 1:
            com = reshape_vector(com, (self.num_humanoids, 1, -1))
            x_i_pos = reshape_vector(x_i.pos, (self.num_humanoids, -1, 3))
            pos_replace = reshape_vector(self._flatten(x_i_pos - com), (-1, 3))
            cinr = x_i.replace(pos=pos_replace).vmap().do(inertia)
            mass_sum = self._flatten(
                mass_sum[0]
            )  # double check that mass_sum arent different btw the two robots

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
        if (self.num_humanoids) == 1:
            mass_sum = jp.sum(inertia.mass)
            x_i = pipeline_state.x.vmap().do(inertia.transform)
            com = (
                jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
            )
        else:
            inertia_mass = reshape_vector(inertia.mass, (self.num_humanoids, -1, 1))
            mass_sum = jp.sum(inertia_mass, axis=1)
            x_i = pipeline_state.x.vmap().do(inertia.transform)
            x_i_pos = reshape_vector(x_i.pos, (self.num_humanoids, -1, 3))
            com = jp.sum(
                jax.vmap(jp.multiply)(inertia_mass, x_i_pos), axis=1
            ) / reshape_vector(mass_sum, (-1, 1))
        return com, inertia, mass_sum, x_i

    @property
    def dims(self):
        action_dim = int(self.sys.act_size() / self.num_humanoids)
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
        return 17 * self.num_humanoids


@ft.partial(jax.jit, static_argnums=1)
def reshape_vector(vector, target_shape):
    return jp.reshape(vector, target_shape)


if __name__ == "__main__":
    # ===============Config===============
    device = "cuda"
    num_envs = 2048
    episode_length = 1000
    # ===============Config===============
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

    action = (
        torch.ones((env.action_space.shape[0], env.action_space.shape[1] * 2)).to(
            device
        )
        * 2
    )
    env.step(action)
