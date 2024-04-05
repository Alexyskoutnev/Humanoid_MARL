"""Environments for training and evaluating policies."""

import functools
from typing import Optional, Type

from Humanoid_MARL.envs import (
    humanoids,
    humanoid,
    humanoids_wall,
    ants,
    point_mass_env,
    simple_robot,
)
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers import training
from Humanoid_MARL.envs.wrappers import VmapWrapper, EpisodeWrapper
import jax

_envs = {
    "humanoids": humanoids.Humanoid,
    "humanoids_debug": humanoids.Humanoid,
    "humanoid": humanoid.Humanoid,
    "humanoids_wall": humanoids_wall.Humanoid,
    "humanoids_wall_debug": humanoids_wall.Humanoid,
    "ants": ants.Ants,
    "ants_debug": ants.Ants,
    "point_mass": point_mass_env.Point_Mass,
    "simple_robots": simple_robot.Simple_Robot,
    "simple_robots_debug": simple_robot.Simple_Robot,
}


def get_environment(env_name: str, **kwargs) -> Env:
    """Returns an environment from the environment registry.

    Args:
      env_name: environment name string
      **kwargs: keyword arguments that get passed to the Env class constructor

    Returns:
      env: an environment
    """
    return _envs[env_name](**kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
    """Adds an environment to the registry.

    Args:
      env_name: environment name string
      env_class: the Env class to add to the registry
    """
    _envs[env_name] = env_class


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    device_idx: int = 0,
    **kwargs,
) -> Env:
    """Creates an environment from the registry.

    Args:
      env_name: environment name string
      episode_length: length of episode
      action_repeat: how many repeated actions to take per environment step
      auto_reset: whether to auto reset the environment after an episode is done
      batch_size: the number of environments to batch together
      **kwargs: keyword argments that get passed to the Env class constructor

    Returns:
      env: an environment
    """
    env = _envs[env_name](**kwargs)
    if device_idx == None:
        device_name = "cpu"
    else:
        try:
            device_name = jax.local_devices()[int(device_idx)]
        except:
            device_name = jax.local_devices()[0]

    if episode_length is not None:
        env = training.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = VmapWrapper(env, batch_size, device=device_name)
    if auto_reset:
        env = training.AutoResetWrapper(env)

    return env
