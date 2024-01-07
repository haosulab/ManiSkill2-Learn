import time

import numpy as np
from maniskill2_learn.utils.data import DictArray, GDict, to_np
from maniskill2_learn.utils.meta import get_logger, get_world_size, get_world_rank

from .builder import ROLLOUTS
from .env_utils import build_vec_env


@ROLLOUTS.register_module()
class Rollout:
    def __init__(self, env_cfg, num_procs=20, with_info=False, seed=None, **kwargs):
        get_logger().info(f"Rollout environments have seed from [{seed}, {seed + num_procs})")
        self.vec_env = build_vec_env(env_cfg, num_procs, seed=seed, **kwargs)
        self.with_info = with_info
        self.num_envs = self.vec_env.num_envs
    
    def __getattr__(self, name):
        return getattr(self.vec_env, name)

    def reset(self, idx=None, *args, **kwargs):
        if idx is not None:
            kwargs = dict(**kwargs, idx=idx)
        return self.vec_env.reset(*args, **kwargs)

    def _process_infos(self, infos):
        assert (
            isinstance(infos, dict) and len(infos) == 8
        ), f"Output of step_dict should have length 7! The info have type {type(infos)} and size {len(infos)}, keys {infos.keys()}!"
        if not self.with_info:
            infos.pop("infos")

    def forward_with_policy(self, pi=None, num=1, on_policy=False, replay=None):
        if pi is None:
            assert not on_policy
            ret = self.vec_env.step_random_actions(num)
            self._process_infos(ret)
            return ret

        sim_time, agent_time, oh_time = 0, 0, 0
        import torch
        from maniskill2_learn.utils.torch import barrier, build_dist_var

        def get_actions(idx=None):
            done_index = self.vec_env.done_idx
            if len(done_index) > 0:
                self.reset(idx=done_index)

            obs = DictArray(self.vec_env.recent_obs).slice(idx, wrapper=False) if idx is not None else self.vec_env.recent_obs
            with torch.no_grad():
                with pi.no_sync(mode="actor"):
                    actions = pi(obs)
                    actions = to_np(actions)
            return actions

        if on_policy:
            assert replay is not None, "Directly save samples to replay buffer to save memory."
            world_size = get_world_size()
            num_done = build_dist_var("num_done", "int")
            trajs = [[] for i in range(self.vec_env.num_envs)]
            total, unfinished, finished, ret = 0, 0, 0, None
            true_array = np.ones(1, dtype=np.bool_)
            last_get_done = time.time()

            while total < num:
                st = time.time()
                actions = get_actions()
                agent_time += time.time() - st

                st = time.time()
                item = self.vec_env.step_dict(actions)
                sim_time += time.time() - st
                self._process_infos(item)

                st = time.time()
                unfinished += self.num_envs
                total += self.num_envs
                item["worker_indices"] = np.arange(self.num_envs, dtype=np.int32)[:, None]
                item["is_truncated"] = np.zeros(self.num_envs, dtype=np.bool_)[:, None]
                item = DictArray(item).copy().to_numpy()
                for i in range(self.num_envs):
                    item_i = item.slice(i)  # Make a safe copy to replay buffer!
                    trajs[i].append(item_i)
                    if item_i["episode_dones"][0]:
                        unfinished -= len(trajs[i])
                        if len(trajs[i]) + finished > num:
                            trajs[i] = trajs[i][: num - finished]

                        replay.push_batch(DictArray.stack(trajs[i], axis=0, wrapper=False))
                        finished += len(trajs[i])
                        trajs[i] = []

                oh_time += time.time() - st
                if total >= num * 0.8 and (time.time() - last_get_done) >= 1:
                    last_get_done = time.time()
                    if num_done.get() >= world_size * 0.5:
                        # Use the trick in DD-PPO
                        break

            st = time.time()
            if unfinished > 0:
                for i in range(self.num_envs):
                    if len(trajs[i]) > 0 and finished < num:
                        if len(trajs[i]) + finished > num:
                            trajs[i] = trajs[i][: num - finished]
                        trajs[i][-1]["is_truncated"] = true_array
                        traj_i = DictArray.stack(trajs[i], axis=0).to_two_dims(False)
                        replay.push_batch(traj_i)
                        finished += len(trajs[i])
                del trajs

            num_done.add(1)
            barrier()
            del num_done
            oh_time += time.time() - st
            get_logger().info(
                f"Finish with {finished} samples, simulation time/FPS:{sim_time:.2f}/{finished / sim_time:.2f}, agent time/FPS:{agent_time:.2f}/{finished / agent_time:.2f}, overhead time:{oh_time:.2f}"
            )

            ret = replay.get_all().memory
        else:
            assert num % self.num_envs == 0, f"{num} % {self.num_envs} != 0, some processes are idle, you are wasting memory!"
            ret = []
            for i in range(num // self.num_envs):
                action = get_actions()
                item = self.vec_env.step_dict(action)
                self._process_infos(item)
                item = GDict(item).to_numpy().copy(wrapper=False)
                ret.append(item)
            ret = DictArray.concat(ret, axis=0).to_array().to_two_dims(False)
            if replay is not None:
                replay.push_batch(ret)
        return ret

    def close(self):
        self.vec_env.close()


@ROLLOUTS.register_module()
class NetworkRollout:
    def __init__(self, model, reward_only=False, use_cost=False, num_samples=4, **kwargs):
        self.reward_only = reward_only
        self.model = model
        self.num_envsum_models = self.model.num_heads
        self.num_envsum_samples = num_samples
        self.is_cost = -1 if use_cost else 1

    def reset(self, **kwargs):
        if hasattr(self.model, "reset"):
            self.model.reset()

    def random_action(self):
        raise NotImplementedError

    def step_states_actions(self, states, actions):
        """
        :param states: [n, m] n different env states
        :param actions: [n, c, a] n sequences of actions
        :return: rewards [n, c, 1]
        """
        assert self.reward_only
        batch_size = actions.shape[0]
        len_seq = actions.shape[1]
        assert states.shape[0] == actions.shape[0]
        import torch

        with torch.no_grad():
            device = self.model.device
            current_states = (
                DictArray(states)
                .to_torch(dtype="float32", device=device, non_blocking=True)
                .unsqueeze(1)
                .repeat(self.num_envsum_models, axis=1)
                .repeat(self.num_envsum_samples, axis=0, wrapper=False)
            )
            actions = (
                DictArray(actions).to_torch(dtype="float32", device=device, non_blocking=True).repeat(self.num_envsum_samples, axis=0, wrapper=False)
            )
            assert current_states.ndim == 3
            rewards = []
            # print(len_seq)
            for i in range(len_seq):
                current_actions = actions[:, i : i + 1].repeat_interleave(self.num_envsum_models, dim=1)
                # print(current_actions.shape)
                # print(current_states.mean(0).mean(0), current_actions.mean(0).mean(0))
                # print(current_states, current_actions)
                # print(current_states.shape, current_actions.shape)
                next_obs, r, done = self.model(current_states, current_actions)
                # print('NO', next_obs)
                # exit(0)
                # print(r.mean())
                # exit(0)
                assert r.ndim == 2 and done.ndim == 2
                current_states = next_obs
                rewards.append(r.mean(dim=1).detach())
            rewards = DictArray.stack(rewards, axis=1).to_numpy(wrapper=False)
            rewards[rewards != rewards] = -1e6

            # print(rewards.sum(-1).mean(), rewards.shape)
            # exit(0)

            rewards = rewards.reshape(batch_size, self.num_envsum_samples, len_seq).mean(1)
        return rewards[..., None]


@ROLLOUTS.register_module()
class OptimizationRollout:
    def __init__(self, env_cfg, **kwargs):
        self.logger = get_logger()
        self.vec_env = build_vec_env(env_cfg)
        self.vec_env.reset()
        x, value = self.vec_env.model.get_global_minimum(self.vec_env.model.d)
        self.logger.info(f"{x} {value}!")
        # print('????')
        # exit(0)
        # self.is_cost = -1 if use_cost else 1

    def __getattr__(self, name):
        return getattr(self.vec_env, name)

    def _get_reward(self, x):
        return self.vec_env.step(x)[1]

    def reset(self, **kwargs):
        return np.zeros(1)

    def random_action(self):
        raise NotImplemented

    def step_states_actions(self, states, actions):
        # states: [N, S]
        # actions: [N, 1, NA]
        assert actions.shape[1] == 1 and actions.ndim == 3
        actions = actions[:, 0]
        # print(actions.shape)
        reward = np.apply_along_axis(self._get_reward, 1, actions) * self.is_cost
        # print(reward.shape)
        return reward[:, None, None]
