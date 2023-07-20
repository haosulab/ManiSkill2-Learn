from distutils.log import info
import numpy as np, time
from gymnasium.core import Env

from maniskill2_learn.utils.data import (
    DictArray,
    GDict,
    SharedDictArray,
    split_list_of_parameters,
    concat,
    is_num,
    decode_np,
    repeat,
    SLICE_ALL,
    is_np_arr,
    is_tuple_of,
    is_list_of,
    index_to_slice
)
from maniskill2_learn.utils.meta import Worker
from maniskill2_learn.utils.math import split_num
from .action_space_utils import stack_action_space
from .env_utils import build_env, get_max_episode_steps, convert_observation_to_space
from .wrappers import ExtendedEnv, BufferAugmentedEnv, ExtendedWrapper

"""
@property
def running_idx(self):
    return [i for i, env in enumerate(self.workers) if env.is_running]

@property
def ready_idx(self):
    return [i for i, env in enumerate(self.workers) if env.is_ready]

@property
def idle_idx(self):
    return [i for i, env in enumerate(self.workers) if env.is_idle]

"""


def create_buffer_for_env(env, num_envs=1, shared_np=True):
    assert isinstance(env, ExtendedEnv)
    obs = env.reset()
    item = [obs, np.float32(1.0), True, True, GDict(env.step(env.action_space.sample())[-1]).to_array().float(), env.render()]
    buffer = DictArray(GDict(item).to_array(), capacity=num_envs)
    if shared_np:
        buffer = SharedDictArray(buffer)
    return buffer


class UnifiedVectorEnvAPI(ExtendedWrapper):
    """
    This wrapper is necessary for all environments. Otherwise some output will be the buffer and you can not use list to store them!!!!
    """
    def __init__(self, vec_env):
        super(UnifiedVectorEnvAPI, self).__init__(vec_env)
        assert isinstance(vec_env, VectorEnvBase), f"Please use correct type of environments {type(vec_env)}!"
        self.vec_env, self.num_envs, self.action_space = vec_env, vec_env.num_envs, vec_env.action_space
        self.all_env_indices = np.arange(self.num_envs, dtype=np.int32)
        self.single_env = self.vec_env.single_env
        self.is_discrete, self.reward_scale, self.is_cost = self.single_env.is_discrete, self.single_env.reward_scale, self.single_env.is_cost

        self.recent_obs = DictArray(self.vec_env.reset(idx=self.all_env_indices)).copy()
        self.episode_dones = np.zeros([self.num_envs, 1], dtype=np.bool_)
        self.dirty = False

    @property
    def done_idx(self):
        return np.nonzero(self.episode_dones)[0]

    def _process_idx(self, idx):
        if idx is None:
            slice_idx = SLICE_ALL
            idx = self.all_env_indices
        else:
            slice_idx = index_to_slice(idx)
        self.vec_env._assert_id(idx)
        return idx, slice_idx

    def reset(self, idx=None, *args, **kwargs):
        self.dirty = False

        idx, slice_idx = self._process_idx(idx)
        args, kwargs = list(args), dict(kwargs)
        if len(args) > 0:
            for arg_i in args:
                if not hasattr(arg_i, '__len__'):
                    arg_i = [arg_i for i in idx]
                else:
                    assert len(arg_i) == len(idx), f"Len of value {len(arg_i)} is not {len(idx)}!"
        if len(kwargs) > 0:
            for key, value in kwargs.items():
                if not hasattr(value, '__len__'):
                    kwargs[key] = [value for i in idx]
                else:
                    assert len(value) == len(idx), f"Len of value {len(value)} is not {len(idx)}!"
        obs = self.vec_env.reset(idx=idx, *args, **kwargs)
        self.episode_dones[slice_idx] = False
        self.recent_obs.assign(slice_idx, obs)
        return GDict(obs).copy(wrapper=False)

    def step(self, actions, idx=None):
        assert not self.dirty, "You need to reset environment after doing step_states_actions!"
        idx, slice_idx = self._process_idx(idx)
        assert len(actions) == len(idx)
        alls = self.vec_env.step(actions, idx=idx)
        self.recent_obs.assign(slice_idx, alls[0])
        self.episode_dones[slice_idx] = np.logical_or(alls[2], alls[3])
        return GDict(alls).copy(wrapper=False)

    def render(self, idx=None):
        return self.vec_env.render(self._process_idx(idx)[0])

    def step_states_actions(self, *args, **kwargs):
        self.vec_env._assert_id(self.all_env_indices)
        self.dirty = True
        return self.vec_env.step_states_actions(*args, **kwargs)

    def step_random_actions(self, num):
        self.vec_env._assert_id(self.all_env_indices)
        return self.vec_env.step_random_actions(num)

    def get_attr(self, name, idx=None):
        return self.vec_env.get_attr(name, self._process_idx(idx)[0])

    def call(self, name, idx=None, *args, **kwargs):
        return self.vec_env.call(name, self._process_idx(idx)[0], *args, **kwargs)

    def get_state(self, idx=None):
        return GDict(self.call("get_state", idx=idx)).copy(wrapper=False)

    def get_obs(self, idx=None):
        return GDict(self.call("get_obs", idx=idx)).copy(wrapper=False)

    def set_state(self, state, idx=None):
        return self.call("set_state", state=state, idx=idx)

    # def seed(self, seed, idx=None):
    #     return self.call("seed", seed=seed, idx=idx)

    def get_env_state(self, idx=None):
        return self.call("get_env_state", idx=idx)

    # def random_action(self):
    #     return self.action_space.sample()

    # Special functions
    def step_dict(self, actions, idx=None, restart=True):
        idx, slice_idx = self._process_idx(idx)
        obs = self.recent_obs.slice(slice_idx).copy(wrapper=False)
        next_obs, reward, terminated, truncated, info = self.step(actions, idx=idx)
        term_or_trunc = np.logical_or(terminated, truncated)
        if np.any(term_or_trunc) and restart:
            self.reset(idx=np.where(term_or_trunc[..., 0])[0])
            
        return dict(
            obs=obs,
            next_obs=next_obs,
            actions=GDict(actions).f64_to_f32(wrapper=False),
            rewards=reward,
            dones=terminated,
            episode_dones=term_or_trunc,
            infos=info,
            worker_indices=idx,
        )

    def __getattr__(self, name, idx=None):
        return self.vec_env.get_attr(name, self._process_idx(idx)[0])

    def close(self):
        self.vec_env.close()


class VectorEnvBase(Env):
    SHARED_NP_BUFFER: bool

    def __init__(self, env_cfgs=None, wait_num=None, timeout=None, **kwargs):
        super(VectorEnvBase, self).__init__()
        self.env_cfgs, self.single_env, self.num_envs = env_cfgs, build_env(env_cfgs[0]), len(env_cfgs)

        assert wait_num is None and timeout is None, "We do not support partial env step now!"
        self.timeout = int(1e9) if timeout is None else timeout
        self.wait_num = len(env_cfgs) if wait_num is None and env_cfgs is not None else wait_num
        self.workers, self.buffers = None, None
        self.action_space = stack_action_space(self.single_env.action_space, self.num_envs)
        if self.SHARED_NP_BUFFER is not None:
            self.buffers = create_buffer_for_env(self.single_env, self.num_envs, self.SHARED_NP_BUFFER)
            buffers = self.buffers.memory
            self.reset_buffer = DictArray(buffers[0])
            self.step_buffer = DictArray(buffers[:5])
            self.vis_img_buffer = DictArray(buffers[5])

    def __getattr__(self, name, idx=None):
        if not hasattr(self.single_env, name):
            return None
        else:
            return self.get_attr(name, idx)
            
    def _init_obs_space(self):
        self.observation_space = convert_observation_to_space(self.reset(idx=np.arange(self.num_envs)))

    def _assert_id(self, idx=None):
        raise NotImplementedError

    def reset(self, idx=None, *args, **kwargs):
        raise NotImplementedError

    def step(self, actions, idx=None):
        raise NotImplementedError

    def render(self, idx=None):
        raise NotImplementedError

    def step_states_actions(self, states, actions):
        raise NotImplementedError

    def step_random_actions(self, num):
        raise NotImplementedError

    def get_attr(self, name, idx=None):
        raise NotImplementedError

    def call(self, name, idx=None, *args, **kwargs):
        raise NotImplementedError

    def get_obs(self, idx=None):
        return self.call("get_obs", idx)

    def get_state(self, idx=None):
        return self.call("get_state", idx)

    def set_state(self, state, idx=None):
        return self.call("set_state", state=state, idx=idx)

    def get_env_state(self, idx=None):
        return self.call("get_env_state", idx=idx)

    # def seed(self, seed, idx=None):
    #     raise NotImplementedError

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.single_env)

    def close(self):
        if self.workers is not None:
            for worker in self.workers:
                worker.close()
        if self.buffers is not None:
            del self.buffers


class SingleEnv2VecEnv(VectorEnvBase):
    """
    Build vectorized api for single environment!
    """

    SHARED_NP_BUFFER = None

    def __init__(self, cfgs, seed=None, **kwargs):
        assert len(cfgs) == 1
        super(SingleEnv2VecEnv, self).__init__(cfgs, **kwargs)
        base_seed = np.random.randint(int(1e9)) if seed is None else seed
        self._env = self.single_env
        # self._env.seed(base_seed) # seed is removed in gymnasium
        self._init_obs_space()

    def _assert_id(self, idx):
        return True

    def _unsqueeze(self, item):
        if item is not None:
            return GDict(item).to_array().unsqueeze(axis=0, wrapper=False)

    def reset(self, idx=None, *args, **kwargs):
        args = list(args)
        kwargs = dict(kwargs)
        args, kwargs = GDict([args, kwargs]).slice(0, wrapper=False)
        return self._unsqueeze(self._env.reset(*args, **kwargs))

    def step(self, actions, idx=None):
        return self._unsqueeze(self._env.step(actions[0]))

    def render(self, idx=None):
        return self._unsqueeze(self._env.render())

    def step_states_actions(self, *args, **kwargs):
        self.dirty = True
        return self._env.step_states_actions(*args, **kwargs)

    def step_random_actions(self, num):
        ret = self._env.step_random_actions(num)
        ret["worker_indices"] = np.zeros(ret["dones"].shape, dtype=np.int32)
        return GDict(ret).to_two_dims(wrapper=False)

    def get_attr(self, name, idx=None):
        return self._unsqueeze(getattr(self._env, name))

    def call(self, name, idx=None, *args, **kwargs):
        args, kwargs = GDict(list(args)).squeeze(0, False), GDict(dict(kwargs)).squeeze(0, False)
        ret = getattr(self._env, name)(*args, **kwargs)
        ret = GDict(ret).to_array()
        return self._unsqueeze(ret)

    # def seed(self, seed, idx=None):
    #     base_idx = np.random.RandomState(seed).randint(0, int(1E9))
    #     self._env.seed(base_idx)


class VectorEnv(VectorEnvBase):
    """
    Always use shared memory and requires the environment have type BufferAugmentedEnv
    """

    SHARED_NP_BUFFER = True

    def __init__(self, env_cfgs, seed=None, **kwargs):
        super(VectorEnv, self).__init__(env_cfgs=env_cfgs, **kwargs)

        base_seed = np.random.randint(int(1e9)) if seed is None else seed
        self.workers = [Worker(build_env, i, base_seed + i, True, self.buffers.get_infos(), cfg=cfg) for i, cfg in enumerate(env_cfgs)]
        self._init_obs_space()

    def _assert_id(self, idx):
        for i in idx:
            assert self.workers[i].is_idle, f"Cannot interact with environment {i} which is stepping now."

    def reset(self, idx=None, *args, **kwargs):
        args, kwargs = list(args), dict(kwargs)
        all_kwargs = GDict([args, kwargs])
        for i in range(len(idx)):
            args_i, kwargs_i = all_kwargs.slice(i, wrapper=False)
            self.workers[idx[i]].call("reset", *args_i, **kwargs_i)
        [self.workers[i].wait() for i in idx]
        return self.reset_buffer.slice(index_to_slice(idx), wrapper=False)

    def step(self, actions, idx=None):
        slice_idx = slice(None, None, None) if len(idx) == len(self.workers) else idx
        for i in range(len(idx)):
            self.workers[idx[i]].call("step", action=actions[i])
        [self.workers[i].wait() for i in idx]
        return self.step_buffer.slice(slice_idx, wrapper=False)

    def render(self, idx=None):
        for i in idx:
            self.workers[i].call("render")
        [self.workers[i].wait() for i in idx]
        return self.vis_img_buffer.slice(index_to_slice(idx), wrapper=False)

    def step_random_actions(self, num):
        # For replay buffer warmup of the RL agent
        n, num_per_env = split_num(num, self.num_envs)
        self._assert_id(list(range(n)))

        shared_mem_value = [bool(self.workers[i].shared_memory.value) for i in range(n)]
        for i, num_i in enumerate(num_per_env):
            self.workers[i].set_shared_memory(False)
            self.workers[i].call("step_random_actions", num=num_i)

        ret = []
        for i in range(n):
            ret_i = self.workers[i].wait()
            ret_i["worker_indices"] = np.ones(ret_i["dones"].shape, dtype=np.int32) * i
            ret.append(ret_i)
            self.workers[i].set_shared_memory(shared_mem_value[i])
        return DictArray.concat(ret, axis=0, wrapper=False)

    def step_states_actions(self, states, actions):
        """
        Return shape: [N, LEN, 1]
        """
        # For MPC
        self.dirty = True

        paras = split_list_of_parameters(self.num_envs, states=states, actions=actions)
        n = len(paras)
        self._assert_id(list(range(n)))

        shared_mem_value = [bool(self.workers[i].shared_memory.value) for i in range(n)]
        for i in range(n):
            args_i, kwargs_i = paras[i]
            self.workers[i].set_shared_memory(False)
            self.workers[i].call("step_states_actions", *args_i, **kwargs_i)
        ret = []
        for i in range(n):
            ret.append(self.workers[i].wait())
            self.workers[i].set_shared_memory(shared_mem_value[i])
        return concat(ret, axis=0)

    def get_attr(self, name, idx=None):
        shared_mem_value = [bool(self.workers[i].shared_memory.value) for i in idx]
        for i in idx:
            self.workers[i].set_shared_memory(False)
            self.workers[i].get_attr(name)
        ret = []
        for i, mem_flag in zip(idx, shared_mem_value):
            ret.append(self.workers[i].wait())
            self.workers[i].set_shared_memory(mem_flag)
        ret = GDict(ret).to_array()
        return GDict.stack(ret, axis=0, wrapper=False)

    def call(self, name, idx=None, *args, **kwargs):
        args, kwargs = GDict(list(args)), GDict(dict(kwargs))

        shared_mem_value = [bool(self.workers[i].shared_memory.value) for i in idx]
        for j, i in enumerate(idx):
            self.workers[i].set_shared_memory(False)
            self.workers[i].call(name, *args.slice(j, 0, False), **kwargs.slice(j, 0, False))

        ret = []
        for i, mem_flag in zip(idx, shared_mem_value):
            ret.append(self.workers[i].wait())
            self.workers[i].set_shared_memory(mem_flag)
        ret = GDict(ret).to_array()
        return None if ret[0] is None else GDict.stack(ret, axis=0, wrapper=False)

    # def seed(self, seed, idx=None):
    #     base_idx = np.random.RandomState(seed).randint(0, int(1E9))
    #     [self.workers[i].call("seed", base_idx + i) for i in idx]


class SapienThreadEnv(VectorEnvBase):
    SHARED_NP_BUFFER = False

    # This vectorized env is designed for maniskill
    def __init__(self, env_cfgs, seed=None, **kwargs):
        self._check_cfgs(env_cfgs)
        super(SapienThreadEnv, self).__init__(env_cfgs, **kwargs)
        self.workers = []
        for i, cfg in enumerate(env_cfgs):
            cfg["buffers"] = self.buffers.slice(i, wrapper=False)
            self.workers.append(build_env(cfg))

        base_seed = np.random.randint(int(1e9)) if seed is None else seed
        # [env.seed(base_seed + i) for i, env in enumerate(self.workers)] # seed is removed in gymnasium

        # For step async in sapien
        self._num_finished = 0
        self._env_indices = np.arange(self.num_envs)
        # -1: idle, 0: simulation, 1: rendering, 2: done (need to be reset to idle after get all information)
        self._env_stages = np.ones(self.num_envs, dtype=np.int32) * -1
        self._env_flags = [None for i in range(self.num_envs)]
        self._init_obs_space()

    @classmethod
    def _check_cfgs(self, env_cfgs):
        sign = True
        for cfg in env_cfgs:
            sign = sign and (cfg.get("with_torch", False) and cfg.get("with_cpp", False))
        if not sign:
            from maniskill2_learn.utils.meta import get_logger

            logger = get_logger()
            logger.warning("You need to use torch and cpp extension, otherwise the speed is not fast enough!")

    def _assert_id(self, idx):
        for i in idx:
            assert self._env_stages[i] == -1, f"Cannot interact with environment {i} which is stepping now."

    def reset(self, level=None, idx=None):
        self._env_stages[idx] = -1

        for i in range(len(idx)):
            self.workers[idx[i]].reset_no_render(level if is_num(level) or level is None else level[i])
        for i in range(len(idx)):
            self.workers[idx[i]].get_obs(sync=False)
        for i in range(len(idx)):
            self.workers[idx[i]].image_wait(mode='o')
        return self.reset_buffer.slice(index_to_slice(idx), wrapper=False)

    def step(self, actions, idx=None, rew_only=False):
        import sapien
        wait_num = len(idx)

        with sapien.core.ProfilerBlock("step_async"):
            self._num_finished = 0
            self._env_stages[idx] = 0
            for i in range(len(idx)):
                self._env_flags[idx[i]] = self.workers[idx[i]].step_async(actions[i])

        with sapien.core.ProfilerBlock("call render-async"):
            render_jobs = []
            while self._num_finished < wait_num:
                for i in range(self.num_envs):
                    if self._env_stages[i] == 0 and self._env_flags[i].ready():
                        self._env_stages[i] += 1
                        self._num_finished += 1
                        if not rew_only:
                            self.workers[i].call_renderer_async(mode="o")
                            render_jobs.append(i)
    
        for i in render_jobs:
            self.workers[i].get_obs(sync=False)
        with sapien.core.ProfilerBlock("wait for render"):
            for i in render_jobs:
                self.workers[i].image_wait(mode='o')
            self._env_stages[idx] = -1

        if rew_only:
            return self.step_buffer[1][idx]

        alls = self.step_buffer.slice(index_to_slice(idx), wrapper=False)
        alls[-1] = self.single_env.deserialize_info(alls[-1])
        return alls

    def render(self, idx=None):
        if self.workers[0].render_mode == "human":
            assert len(self.workers) == 1, "Human rendering only allows num_envs = 1!"
            return self.workers[0].render()
        assert self.workers[0].render_mode == "rgb_array", "We only support rgb_array mode for multiple environments!"
        [self.workers[i].call_renderer_async(mode="v") for i in idx]
        [self.workers[i].image_wait(mode="v") for i in idx]
        return self.vis_img_buffer.slice(index_to_slice(idx), wrapper=False)

    def step_random_actions(self, num):
        # For replay buffer warmup of the RL agent

        obs = self.reset(idx=np.arange(self.num_envs))
        num = int(num)
        ret = []
        while num > 0:
            num_i = min(num, self.num_envs)
            actions = self.action_space.sample()[:num_i]
            idx = np.arange(num_i, dtype=np.int32)
            next_obs, rewards, terminates, truncates, infos = self.step(actions, idx=idx)
            term_or_truncs = np.logical_or(terminates, truncates)
            ret_i = dict(
                obs=obs,
                next_obs=next_obs,
                actions=actions,
                rewards=rewards,
                dones=terminates,
                infos=infos,
                episode_dones=term_or_truncs,
                worker_indices=idx[:, None],
            )
            ret.append(GDict(ret_i).to_array().copy(wrapper=False))
            obs = GDict(next_obs).copy(wrapper=False)
            num -= num_i
            if np.any(dones):
                self.reset(idx=np.where(dones[..., 0])[0])
        return DictArray.concat(ret, axis=0).to_two_dims(wrapper=False)

    def step_states_actions(self, states=None, actions=None):
        """
        Return shape: [N, LEN, 1]
        """
        # For MPC
        rewards = np.zeros_like(actions[..., :1], dtype=np.float32)
        for i in range(0, len(actions), self.num_envs):
            num_i = min(len(actions) - i, self.num_envs)
            if hasattr(self, "set_state") and states is not None:
                for j in range(num_i):
                    self.workers[j].set_state(states[i + j])

            for j in range(len(actions[i])):
                rewards[i : i + num_i, j] = self.step(actions[i : i + num_i, j], idx=np.arange(num_i), rew_only=True)
        return rewards

    def get_attr(self, name, idx=None):
        ret = GDict([getattr(self.workers[i], name) for i in idx]).to_array()
        return GDict.stack(ret, 0, wrapper=False)

    def call(self, name, idx=None, *args, **kwargs):
        args, kwargs = GDict(list(args)), GDict(dict(kwargs))
        ret = [getattr(self.workers[i], name)(*args.slice(i, 0, False), **kwargs.slice(i, 0, False)) for i in idx]
        ret = GDict(ret).to_array()
        return None if ret[0] is None else GDict.stack(ret, axis=0, wrapper=False)
