from collections import OrderedDict
import gym, numpy as np
from gym.envs import registry
from gym.spaces import Box, Discrete, Dict
from gym.wrappers import TimeLimit
from maniskill2_learn.utils.data import GDict, get_dtype
from maniskill2_learn.utils.meta import Registry, build_from_cfg, dict_of, get_logger
from .action_space_utils import StackedDiscrete, unstack_action_space
from .wrappers import ManiSkill2_ObsWrapper, RenderInfoWrapper, ExtendedEnv, BufferAugmentedEnv, build_wrapper

ENVS = Registry("env")


def import_env():
    import contextlib
    import os
    import mani_skill2.envs


def convert_observation_to_space(observation):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from gym.envs.mujoco_env
    """
    if isinstance(observation, (dict)):
        # if not isinstance(observation, OrderedDict):
        #     warn("observation is not an OrderedDict. Keys are {}".format(observation.keys()))
        space = Dict(OrderedDict([(key, convert_observation_to_space(value)) for key, value in observation.items()]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=observation.dtype)
        high = np.full(observation.shape, float("inf"), dtype=observation.dtype)
        space = Box(low, high, dtype=observation.dtype)
    else:
        import torch

        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
            return convert_observation_to_space(observation)
        else:
            raise NotImplementedError(type(observation), observation)

    return space


def get_gym_env_type(env_name):
    import_env()
    if env_name not in registry.env_specs:
        raise ValueError("No such env")
    try:
        entry_point = registry.env_specs[env_name].entry_point
        if entry_point.startswith("gym.envs."):
            type_name = entry_point[len("gym.envs.") :].split(":")[0].split(".")[0]
        else:
            type_name = entry_point.split(".")[0]
    except AttributeError as e:
        # ManiSkill2
        entry_point = registry.env_specs[env_name].entry_point.func.__code__.co_filename
        if 'mani_skill2' in entry_point:
            type_name = 'mani_skill2'
        else:
            print("Can't process the entry point: ", entry_point)
            raise e

    return type_name


def true_done(done, info):
    if "TimeLimit.truncated" not in info:  # default is True!
        return False

    time_truncated = info["TimeLimit.truncated"]
    if isinstance(done, (bool, np.bool_)):
        return False if time_truncated else done
    else:
        if get_dtype(time_truncated) in ["float32", "float64"]:
            time_truncated = time_truncated > 0.5
        return np.logical_and(done, ~time_truncated)


def get_env_info(env_cfg=None, vec_env=None):
    """
    For observation space, we use obs_shape instead of gym observation space which is not easy to use in network building!
    """
    vec_env = build_vec_env(env_cfg.copy()) if vec_env is None else vec_env
    obs_shape = GDict(vec_env.reset()).slice(0).list_shape
    action_space = unstack_action_space(vec_env.action_space)
    action = action_space.sample()
    assert isinstance(action_space, (Box, StackedDiscrete)), f"Error type {type(action_space)}!"
    is_discrete = isinstance(action_space, StackedDiscrete)
    if is_discrete:
        action_shape = vec_env.action_space.n
        get_logger().info(f"Environment has the discrete action space with {action_shape} choices.")
    else:
        action_shape = action.shape[0]
        get_logger().info(f"Environment has the continuous action space with dimension {action_shape}.")
    del vec_env
    return dict_of(obs_shape, action_shape, action_space, is_discrete)


def get_max_episode_steps(env):
    if hasattr(env, "_max_episode_steps"):
        return env._max_episode_steps
    elif hasattr(env.unwrapped, "_max_episode_steps"):
        # For env that does not use TimeLimit, e.g. ManiSkill
        return env.unwrapped._max_episode_steps
    else:
        raise NotImplementedError("Your environment needs to contain the attribute _max_episode_steps!")


def make_gym_env(
    env_name,
    unwrapped=False,
    horizon=None,
    time_horizon_factor=1,
    stack_frame=1,
    use_cost=False,
    reward_scale=1,
    worker_id=None,
    buffers=None,
    **kwargs,
):
    """
    If we want to add custom wrapper, we need to unwrap the env if the original env is wrapped by TimeLimit wrapper.
    All environments will have ExtendedTransformReward && SerializedInfoEnv outside, also the info dict is always serailzed!
    """

    import_env()
    kwargs = dict(kwargs)
    kwargs.pop("multi_thread", None)
    env_type = get_gym_env_type(env_name)
    if env_type not in ["mani_skill2",]:
        # For environments that cannot specify GPU, we pop device
        kwargs.pop("device", None)

    # Sapien callback system use buffer by default
    if env_type in "maniskill":
        kwargs["buffers"] = buffers
        buffers = None

    extra_wrappers = kwargs.pop("extra_wrappers", None)
    if env_type == "mani_skill2":
        # Extra kwargs for maniskill2
        img_size = kwargs.pop("img_size", None)
        n_points = kwargs.pop("n_points", 1200)
        n_goal_points = kwargs.pop("n_goal_points", -1)
        obs_frame = kwargs.pop('obs_frame', 'world')
        ignore_dones = kwargs.pop('ignore_dones', False)
        fix_seed = kwargs.pop("fix_seed", None)

    env = gym.make(env_name, **kwargs)

    if env is None:
        print(f"No {env_name} in gym")
        exit(0)

    use_time_limit = False
    max_episode_steps = get_max_episode_steps(env) if horizon is None else int(horizon)
    if isinstance(env, TimeLimit):
        env = env.env
        use_time_limit = True
    elif hasattr(env.unwrapped, "_max_episode_steps"):
        if horizon is not None:
            env.unwrapped._max_episode_steps = int(horizon)
        else:
            env.unwrapped._max_episode_steps = int(max_episode_steps * time_horizon_factor)

    if unwrapped:
        env = env.unwrapped if hasattr(env, "unwrapped") else env

    if env_type == "mani_skill2":
        env = RenderInfoWrapper(env)
        env = ManiSkill2_ObsWrapper(env, img_size=img_size, 
            n_points=n_points, n_goal_points=n_goal_points, obs_frame=obs_frame, 
            ignore_dones=ignore_dones, fix_seed=fix_seed)

    if extra_wrappers is not None:
        if not isinstance(extra_wrappers, list):
            extra_wrappers = [
                extra_wrappers,
            ]
        for extra_wrapper in extra_wrappers:
            extra_wrapper.env = env
            extra_wrapper.env_name = env_name
            env = build_wrapper(extra_wrapper)

    if use_time_limit and not unwrapped:
        env = TimeLimit(env, int(max_episode_steps * time_horizon_factor))

    env = ExtendedEnv(env, reward_scale, use_cost)

    if buffers is not None:
        env = BufferAugmentedEnv(env, buffers=buffers)
    return env


ENVS.register_module("gym", module=make_gym_env)


def build_env(cfg, **kwargs):
    cfg.update(**kwargs)
    return build_from_cfg(cfg, ENVS)


def build_vec_env(cfgs, num_procs=None, multi_thread=False, **vec_env_kwargs):
    # add a wrapper to gym.make to make the environment building process more flexible.

    import_env()
    num_procs = num_procs or 1

    if isinstance(cfgs, dict):
        cfgs = [cfgs] * num_procs
    env_type = get_gym_env_type(cfgs[0].env_name)
    assert len(cfgs) == num_procs, "You need to provide env configurations for each process or thread!"

    from .vec_env import VectorEnv, SapienThreadEnv, SingleEnv2VecEnv, UnifiedVectorEnvAPI

    if multi_thread:
        vec_env = SapienThreadEnv(cfgs, **vec_env_kwargs)
    elif len(cfgs) == 1:
        vec_env = SingleEnv2VecEnv(cfgs, **vec_env_kwargs)
    else:
        vec_env = VectorEnv(cfgs, **vec_env_kwargs)

    vec_env = UnifiedVectorEnvAPI(vec_env)
    return vec_env
