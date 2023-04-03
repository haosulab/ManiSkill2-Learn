import copy
import glob
import logging
import os
import os.path as osp
import shutil
from copy import deepcopy

import cv2
import numpy as np
from h5py import File
from maniskill2_learn.utils.data import (
    DictArray,
    GDict,
    concat_list,
    decode_np,
    dict_to_str,
    is_str,
    num_to_str,
    split_list_of_parameters,
    to_np,
    dict_to_str,
    to_item,
)
from maniskill2_learn.utils.file import dump, load, merge_h5_trajectory
from maniskill2_learn.utils.math import split_num
from maniskill2_learn.utils.meta import TqdmToLogger, Worker, get_dist_info, get_logger, get_logger_name, get_total_memory, get_meta_info

from .builder import EVALUATIONS
from .env_utils import build_vec_env, build_env, true_done, get_max_episode_steps
from .replay_buffer import ReplayMemory


def save_eval_statistics(folder, lengths, rewards, finishes, logger=None):
    if logger is None:
        logger = get_logger()
    logger.info(
        f"Num of trails: {len(lengths):.2f}, "
        f"Length: {np.mean(lengths):.2f}\u00B1{np.std(lengths):.2f}, "
        f"Reward: {np.mean(rewards):.2f}\u00B1{np.std(rewards):.2f}, "
        f"Success or Early Stop Rate: {np.mean(finishes):.2f}\u00B1{np.std(finishes):.2f}"
    )
    if folder is not None:
        table = [["length", "reward", "finish"]]
        table += [[num_to_str(__, precision=2) for __ in _] for _ in zip(lengths, rewards, finishes)]
        dump(table, osp.join(folder, "statistics.csv"))


CV_VIDEO_CODES = {
    "mp4": cv2.VideoWriter_fourcc(*"mp4v"),
}


def log_mem_info(logger):
    import torch
    from maniskill2_learn.utils.torch import get_cuda_info

    print_dict = {}
    print_dict["memory"] = get_total_memory("G", False)
    print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))
    print_info = dict_to_str(print_dict)
    logger.info(f"Resource usage: {print_info}")


@EVALUATIONS.register_module()
class FastEvaluation:
    def __init__(self, env_cfg=None, num_procs=1, seed=None, **kwargs):
        self.n = num_procs
        self.vec_env = build_vec_env(env_cfg, num_procs, **kwargs, seed=seed)
        self.vec_env.reset()

        self.num_envs = self.vec_env.num_envs
        self.all_env_indices = np.arange(self.num_envs, dtype=np.int32)
        self.log_every_episode = kwargs.get("log_every_episode", True)
        self.log_every_step = kwargs.get("log_every_step", False)

        self.save_traj = kwargs.get("save_traj", False)
        self.save_video = kwargs.get("save_video", False)
        self.only_save_success_traj = kwargs.get("only_save_success_traj", False)

        self.sample_mode = kwargs.get("sample_mode", "eval")
        self.eval_levels = kwargs.get("eval_levels", None)

        self.video_format = kwargs.get("video_format", "mp4")
        self.video_fps = kwargs.get("fps", 20)

        logger_name = get_logger_name()
        self.logger = get_logger("Evaluation-" + logger_name, with_stream=True)
        self.logger.info(f"Evaluation environments have seed in [{seed}, {seed + num_procs})!")

        if self.eval_levels is not None and is_str(self.eval_levels):
            is_csv = eval_levels.split(".")[-1] == "csv"
            eval_levels = load(self.eval_levels)
            self.eval_levels = eval_levels[0] if is_csv else eval_levels
        if self.eval_levels is not None:
            self.logger.info(f"During evaluation, levels are selected from an existing list with length {len(self.eval_levels)}")

    def reset_pi(self, pi, idx):
        """ When we run CEM, we need the level of the rollout env to match the level of test env.  """
        if not hasattr(pi, "reset"):
            return
        reset_kwargs = {}
        if hasattr(self.vec_env.vec_env.single_env, 'level'):
            reset_kwargs["level"] = self.vec_env.level
        pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

    def run(self, pi, num=1, work_dir=None, **kwargs):
        eval_levels = None
        if self.eval_levels is not None:
            num = min(len(self.eval_levels), num)
            random_start = np.random.randint(len(self.eval_levels) - num + 1)
            eval_levels = self.eval_levels[random_start : random_start + num]
        self.logger.info(f"We will evaluate over {num} episodes!")

        if osp.exists(work_dir):
            self.logger.warning(f"We will overwrite this folder {work_dir} during evaluation!")
            shutil.rmtree(work_dir, ignore_errors=True)
        os.makedirs(work_dir, exist_ok=True)

        if self.save_video:
            video_dir = osp.join(work_dir, "videos")
            self.logger.info(f"Save videos to {video_dir}.")
            os.makedirs(video_dir, exist_ok=True)

        if self.save_traj:
            trajectory_path = osp.join(work_dir, "trajectory.h5")
            if osp.exists(trajectory_path):
                self.logger.warning(f"We will overwrite this file {trajectory_path} during evaluation!")
            h5_file = File(trajectory_path, "w")
            self.logger.info(f"Save trajectory at {trajectory_path}.")
            group = h5_file.create_group(f"meta")
            GDict(get_meta_info()).to_hdf5(group)

        import torch

        num_finished, num_start, num_envs = 0, 0, min(self.num_envs, num)
        traj_idx = np.arange(num_envs, dtype=np.int32)
        video_writers, episodes = None, None

        if eval_levels is not None and hasattr(self.vec_env.vec_env.single_env, 'level'):
            obs_all = self.vec_env.reset(level=eval_levels[:num_envs], idx=np.arange(num_envs))
        else:
            obs_all = self.vec_env.reset(idx=np.arange(num_envs))
        obs_all = DictArray(obs_all).copy()
        self.reset_pi(pi, self.all_env_indices)

        if self.save_video:
            video_writers = []
            imgs = self.vec_env.render(mode="rgb_array", idx=np.arange(num_envs))[..., ::-1]
            for i in range(num_envs):
                video_file = osp.join(video_dir, f"{i}.{self.video_format}")
                video_writers.append(
                    cv2.VideoWriter(video_file, CV_VIDEO_CODES[self.video_format], self.video_fps, (imgs[i].shape[1], imgs[i].shape[0]))
                )
        episodes = [[] for i in range(num_envs)]
        num_start = num_envs
        episode_lens, episode_rewards, episode_finishes = np.zeros(num, dtype=np.int32), np.zeros(num, dtype=np.float32), np.zeros(num, dtype=np.bool_)
        while num_finished < num:
            idx = np.nonzero(traj_idx >= 0)[0]
            obs = obs_all.slice(idx, wrapper=False)
            with torch.no_grad():
                with pi.no_sync(mode="actor"):
                    action = pi(obs, mode=self.sample_mode)
                    action = to_np(action)

            env_state = self.vec_env.get_env_state()
            infos = self.vec_env.step_dict(action, idx=idx, restart=False)
            next_env_state = self.vec_env.get_env_state()
            for key in next_env_state:
                env_state["next_" + key] = next_env_state[key]
            infos.update(env_state)

            infos = GDict(infos).to_array().to_two_dims()
            episode_dones = infos["episode_dones"]
            obs_all.assign(idx, infos["next_obs"])

            if self.log_every_step and self.num_envs == 1:
                reward, done, info, episode_done = GDict([infos["rewards"], infos["dones"], infos["infos"], infos["episode_dones"]]).item(
                    wrapper=False
                )
                assert isinstance(info, dict)
                info_str = dict_to_str({key.split('/')[-1]: val for key, val in info.items()})
                self.logger.info(
                    f"Episode {traj_idx[0]}, Step {episode_lens[traj_idx[0]]}: Reward: {reward:.3f}, Early Stop or Finish: {done}, Info: {info_str}"
                )
            if self.save_video:
                imgs = self.vec_env.render(mode="rgb_array", idx=idx)[..., ::-1]
                for j, i in enumerate(idx):
                    video_writers[i].write(imgs[j])
            reset_idx = []
            reset_levels = []
            for j, i in enumerate(idx):
                episodes[i].append(GDict(infos).slice(j, wrapper=False))
                episode_lens[traj_idx[i]] += 1
                episode_rewards[traj_idx[i]] += to_item(infos["rewards"][j])
                if to_item(episode_dones[j]):
                    num_finished += 1
                    if self.save_video:
                        video_writers[i].release()

                    episodes_i = GDict.stack(episodes[i], 0)
                    episodes[i] = []

                    reward = episodes_i["rewards"].sum()
                    done = to_item(infos["dones"][j])
                    episode_finishes[traj_idx[i]] = done

                    if self.log_every_episode:
                        self.logger.info(
                            f"Episode {traj_idx[i]} ends: Length {episode_lens[traj_idx[i]]}, Reward: {reward}, Early Stop or Finish: {done}!"
                        )
                        log_mem_info(self.logger)

                    if self.save_traj and (not self.only_save_success_traj or done):
                        group = h5_file.create_group(f"traj_{traj_idx[i]}")
                        GDict(episodes_i.memory).to_hdf5(group)

                    if num_start < num:
                        traj_idx[i] = num_start
                        reset_idx.append(i)
                        if eval_levels is not None:
                            reset_levels.append(eval_levels[num_start])
                        num_start += 1
                    else:
                        traj_idx[i] = -1
            reset_idx = np.array(reset_idx, dtype=np.int32)
            if len(reset_idx) > 0:
                if eval_levels is not None:
                    reset_levels = np.array(reset_levels, dtype=np.int64)
                    obs = self.vec_env.reset(level=reset_levels, idx=reset_idx)
                else:
                    obs = self.vec_env.reset(idx=reset_idx)
                obs_all.assign(reset_idx, obs)
                self.reset_pi(pi, reset_idx)

                if self.save_traj:
                    imgs = self.vec_env.render(mode="rgb_array", idx=reset_idx)[..., ::-1]
                    for j, i in enumerate(reset_idx):
                        video_file = osp.join(video_dir, f"{traj_idx[i]}.{self.video_format}")
                        video_writers[i] = cv2.VideoWriter(
                            video_file, CV_VIDEO_CODES[self.video_format], self.video_fps, (imgs[j].shape[1], imgs[j].shape[0])
                        )

        h5_file.close()
        return episode_lens, episode_rewards, episode_finishes

    def close(self):
        self.vec_env.close()


@EVALUATIONS.register_module()
class Evaluation:
    def __init__(
        self,
        env_cfg,
        worker_id=None,
        save_traj=True,
        only_save_success_traj=False,
        save_video=True,
        use_hidden_state=False,
        sample_mode="eval",
        eval_levels=None,
        seed=None,
        **kwargs,
    ):
        
        self.vec_env = build_vec_env(env_cfg, seed=seed)
        self.vec_env.reset()
        self.n = 1

        self.horizon = get_max_episode_steps(self.vec_env.single_env)

        self.save_traj = save_traj
        self.only_save_success_traj = only_save_success_traj
        self.save_video = save_video
        self.vec_env_name = env_cfg.env_name
        self.worker_id = worker_id

        self.video_format = kwargs.get("video_format", "mp4")
        self.video_fps = kwargs.get("fps", 20)
        
        self.log_every_episode = kwargs.get("log_every_episode", True)
        self.log_every_step = kwargs.get("log_every_step", False)

        logger_name = get_logger_name()
        log_level = logging.INFO if (kwargs.get("log_all", False) or self.worker_id is None or self.worker_id == 0) else logging.ERROR
        worker_suffix = "-env" if self.worker_id is None else f"-env-{self.worker_id}"

        self.logger = get_logger("Evaluation-" + logger_name + worker_suffix, log_level=log_level)
        self.logger.info(f"The Evaluation environment has seed in {seed}!")

        self.use_hidden_state = use_hidden_state
        self.sample_mode = sample_mode

        self.work_dir, self.video_dir, self.trajectory_path = None, None, None
        self.h5_file = None

        self.episode_id = 0
        self.level_index = 0
        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        self.recent_obs = None

        self.data_episode = None
        self.video_writer = None
        self.video_file = None

        # restrict the levels as those randomly sampled from eval_levels_path, if eval_levels_path is not None
        if eval_levels is not None:
            if is_str(eval_levels):
                is_csv = eval_levels.split(".")[-1] == "csv"
                eval_levels = load(eval_levels)
                if is_csv:
                    eval_levels = eval_levels[0]
            self.eval_levels = eval_levels
            self.logger.info(f"During evaluation, levels are selected from an existing list with length {len(self.eval_levels)}")
        else:
            self.eval_levels = None

        assert not (self.use_hidden_state and worker_id is not None), "Use hidden state is only for CEM evaluation!!"
        assert self.horizon is not None and self.horizon, f"{self.horizon}"
        assert self.worker_id is None or not use_hidden_state, "Parallel evaluation does not support hidden states!"
        if save_video:
            # Use rendering with use additional 1Gi memory in sapien
            image = self.vec_env.render("rgb_array")[0, ..., ::-1]
            self.logger.info(f"Size of image in the rendered video {image.shape}")

    def start(self, work_dir=None):
        if work_dir is not None:
            self.work_dir = work_dir if self.worker_id is None else os.path.join(work_dir, f"thread_{self.worker_id}")
            # shutil.rmtree(self.work_dir, ignore_errors=True)
            os.makedirs(self.work_dir, exist_ok=True)
            if self.save_video:
                self.video_dir = osp.join(self.work_dir, "videos")
                os.makedirs(self.video_dir, exist_ok=True)
            if self.save_traj:
                self.trajectory_path = osp.join(self.work_dir, "trajectory.h5")
                self.h5_file = File(self.trajectory_path, "w")
                self.logger.info(f"Save trajectory at {self.trajectory_path}.")
                group = self.h5_file.create_group(f"meta")
                GDict(get_meta_info()).to_hdf5(group)

        self.episode_lens, self.episode_rewards, self.episode_finishes = [], [], []
        self.recent_obs = None
        self.data_episode = None
        self.video_writer = None
        self.level_index = -1
        self.logger.info(f"Begin to evaluate in worker {self.worker_id}")

        self.episode_id = -1
        self.reset()

    def done(self):
        self.episode_lens.append(self.episode_len)
        self.episode_rewards.append(self.episode_reward)
        self.episode_finishes.append(self.episode_finish)

        if self.save_traj and self.data_episode is not None:
            if (not self.only_save_success_traj) or (self.only_save_success_traj and self.episode_finish):
                group = self.h5_file.create_group(f"traj_{self.episode_id}")
                self.data_episode.to_hdf5(group, with_traj_index=False)

            self.data_episode = None
        # exit(0)
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def reset(self):
        self.episode_id += 1
        self.episode_len, self.episode_reward, self.episode_finish = 0, 0, False
        level = None
        if self.eval_levels is not None:
            self.level_index = (self.level_index + 1) % len(self.eval_levels)
            # randomly sample a level from the self.eval_init_  levels
            # lvl = int(self.eval_levels[np.random.randint(len(self.eval_levels))])
            # print(f"Env is reset to level {lvl}")
            level = self.eval_levels[self.level_index]
            if isinstance(level, str):
                level = eval(level)
            self.recent_obs = self.vec_env.reset(level=level)
        else:
            self.recent_obs = self.vec_env.reset()
            if hasattr(self.vec_env, "level"):
                level = self.vec_env.level
            elif hasattr(self.vec_env.unwrapped, "_main_seed"):
                level = self.vec_env.unwrapped._main_seed
        if level is not None and self.log_every_episode:
            extra_output = "" if self.level_index is None else f"with level id {self.level_index}"
            self.logger.info(f"Episode {self.episode_id} begins, run on level {level} {extra_output}!")

    def step(self, action):
        data_to_store = {"obs": self.recent_obs}

        if self.save_traj:
            env_state = self.vec_env.get_env_state()
            data_to_store.update(env_state)

        if self.save_video:
            image = self.vec_env.render(mode="rgb_array")[0, ..., ::-1]
            if self.video_writer is None:
                self.video_file = osp.join(self.video_dir, f"{self.episode_id}.{self.video_format}")
                
                self.video_writer = cv2.VideoWriter(
                    self.video_file, CV_VIDEO_CODES[self.video_format], self.video_fps, (image.shape[1], image.shape[0])
                )
            self.video_writer.write(image)
        infos = self.vec_env.step_dict(action, restart=False)

        reward, done, info, episode_done = GDict([infos["rewards"], infos["dones"], infos["infos"], infos["episode_dones"]]).item(wrapper=False)
        self.episode_len += 1
        self.episode_reward += float(reward)
        if self.log_every_step:
            assert isinstance(info, dict)
            info_str = dict_to_str({key.split('/')[-1]: val for key, val in info.items()})
            self.logger.info(
                f"Episode {self.episode_id}, Step {self.episode_len}: Reward: {reward:.3f}, Early Stop or Finish: {done}, Info: {info_str}"
            )

        if self.save_traj:
            data_to_store.update(infos)
            next_env_state = self.vec_env.get_env_state()
            for key in next_env_state:
                data_to_store[f"next_{key}"] = next_env_state[key]
            # if self.data_episode is None: # This trick is problematic for ManiSkill 2022
            if self.data_episode is None:
                self.data_episode = ReplayMemory(self.horizon)
            data_to_store = GDict(data_to_store).to_array().f64_to_f32().to_two_dims()
            self.data_episode.push_batch(data_to_store)

        if episode_done:
            if self.save_video:
                image = self.vec_env.render(mode="rgb_array")[0, ..., ::-1]
                self.video_writer.write(image)
            if self.log_every_episode:
                self.logger.info(
                    f"Episode {self.episode_id} ends: Length {self.episode_len}, Reward: {self.episode_reward}, Early Stop or Finish: {done}"
                )
            self.episode_finish = done
            self.done()
            self.reset()
        else:
            self.recent_obs = infos["next_obs"]
        return self.recent_obs, episode_done

    def finish(self):
        if self.save_traj:
            self.h5_file.close()

    def run(self, pi, num=1, work_dir=None, **kwargs):
        if self.eval_levels is not None:
            if num > len(self.eval_levels):
                print(f"We do not need to select more than {len(self.eval_levels)} levels!")
                num = min(num, len(self.eval_levels))
        self.start(work_dir)
        import torch

        def reset_pi():
            if hasattr(pi, "reset"):
                assert self.worker_id is None, "Reset policy only works for single thread!"
                reset_kwargs = {}
                if hasattr(self.vec_env, "level"):
                    # When we run CEM, we need the level of the rollout env to match the level of test env.
                    reset_kwargs["level"] = self.vec_env.level
                pi.reset(**reset_kwargs)  # For CEM and PETS-like model-based method.

        reset_pi()
        recent_obs = self.recent_obs

        while self.episode_id < num:
            if self.use_hidden_state:
                recent_obs = self.vec_env.get_state()
            with torch.no_grad():
                with pi.no_sync(mode="actor"):
                    action = pi(recent_obs, mode=self.sample_mode)
                    action = to_np(action)
            recent_obs, episode_done = self.step(action)

            if episode_done:
                reset_pi()
                log_mem_info(self.logger)
        self.finish()
        return self.episode_lens, self.episode_rewards, self.episode_finishes

    def close(self):
        if hasattr(self, "env"):
            del self.vec_env
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()


@EVALUATIONS.register_module()
class BatchEvaluation:
    def __init__(
        self,
        env_cfg,
        num_procs=1,
        save_traj=True,
        save_video=True,
        enable_merge=True,
        sample_mode="eval",
        eval_levels=None,
        seed=None,
        **kwargs,
    ):
        self.work_dir = None
        self.vec_env_name = env_cfg.env_name
        self.save_traj = save_traj
        self.save_video = save_video
        self.num_procs = num_procs
        self.enable_merge = enable_merge
        self.sample_mode = sample_mode

        self.video_dir = None
        self.trajectory_path = None
        self.recent_obs = None

        self.n = num_procs
        self.workers = []
        self.logger = get_logger("Evaluation-" + get_logger_name())

        if eval_levels is None:
            eval_levels = [None for i in range(self.n)]
            self.eval_levels = None
        else:
            if is_str(eval_levels):
                is_csv = eval_levels.split(".")[-1] == "csv"
                eval_levels = load(eval_levels)
                if is_csv:
                    eval_levels = eval_levels[0]
            self.eval_levels = eval_levels
            self.n, num_levels = split_num(len(eval_levels), self.n)
            self.logger.info(f"Split {len(eval_levels)} levels into {self.n} processes, and {num_levels}!")
            ret = []
            for i in range(self.n):
                ret.append(eval_levels[: num_levels[i]])
                eval_levels = eval_levels[num_levels[i] :]
            eval_levels = ret
        seed = seed if seed is not None else np.random.randint(int(1E9))
        self.logger.info(f"Evaluation environments have seed in [{seed}, {seed + self.n})!")
        for i in range(self.n):
            self.workers.append(
                Worker(
                    Evaluation,
                    i,
                    worker_seed=seed + i,
                    env_cfg=env_cfg,
                    save_traj=save_traj,
                    seed=seed + i,
                    save_video=save_video,
                    sample_mode=sample_mode,
                    eval_levels=eval_levels[i],
                    **kwargs,
                )
            )

    def start(self, work_dir=None):
        self.work_dir = work_dir
        if self.enable_merge and self.work_dir is not None:
            # shutil.rmtree(self.work_dir, ignore_errors=True)
            self.video_dir = osp.join(self.work_dir, "videos")
            self.trajectory_path = osp.join(self.work_dir, "trajectory.h5")
        for worker in self.workers:
            worker.call("start", work_dir=work_dir)
        for worker in self.workers:
            worker.wait()

        for i in range(self.n):
            self.workers[i].get_attr("recent_obs")
        self.recent_obs = DictArray.concat([self.workers[i].wait() for i in range(self.n)], axis=0)

    @property
    def episode_lens(self):
        for i in range(self.n):
            self.workers[i].get_attr("episode_lens")
        return concat_list([self.workers[i].wait() for i in range(self.n)])

    @property
    def episode_rewards(self):
        for i in range(self.n):
            self.workers[i].get_attr("episode_rewards")
        return concat_list([self.workers[i].wait() for i in range(self.n)])

    @property
    def episode_finishes(self):
        for i in range(self.n):
            self.workers[i].get_attr("episode_finishes")
        return concat_list([self.workers[i].wait() for i in range(self.n)])

    def finish(self):
        for i in range(self.n):
            self.workers[i].call("finish")
        for i in range(self.n):
            self.workers[i].wait()

    def merge_results(self, num_threads):
        if self.save_traj:
            h5_files = [osp.join(self.work_dir, f"thread_{i}", "trajectory.h5") for i in range(num_threads)]
            merge_h5_trajectory(h5_files, self.trajectory_path)
            self.logger.info(f"Merge {len(h5_files)} trajectories to {self.trajectory_path}")
        if self.save_video:
            index = 0
            os.makedirs(self.video_dir)
            for i in range(num_threads):
                num_traj = len(glob.glob(osp.join(self.work_dir, f"thread_{i}", "videos", "*.mp4")))
                for j in range(num_traj):
                    shutil.copyfile(osp.join(self.work_dir, f"thread_{i}", "videos", f"{j}.mp4"), osp.join(self.video_dir, f"{index}.mp4"))
                    index += 1
            self.logger.info(f"Merge {index} videos to {self.video_dir}")
        for dir_name in glob.glob(osp.join(self.work_dir, "*")):
            if osp.isdir(dir_name) and osp.basename(dir_name).startswith("thread"):
                shutil.rmtree(dir_name, ignore_errors=True)

    def run(self, pi, num=1, work_dir=None, **kwargs):
        if self.eval_levels is not None:
            if num > len(self.eval_levels):
                self.logger.info(f"We use number of levels: {len(self.eval_levels)} instead of {num}!")
                num = len(self.eval_levels)

        n, running_steps = split_num(num, self.n)
        self.start(work_dir)
        num_finished = [0 for i in range(n)]
        if hasattr(pi, "reset"):
            pi.reset()
        import torch

        while True:
            finish = True
            for i in range(n):
                finish = finish and (num_finished[i] >= running_steps[i])
            if finish:
                break
            with torch.no_grad():
                with pi.no_sync(mode="actor"):
                    actions = pi(self.recent_obs, mode=self.sample_mode)
                actions = to_np(actions)
            for i in range(n):
                if num_finished[i] < running_steps[i]:
                    self.workers[i].call("step", actions[i:i+1])
            for i in range(n):
                if num_finished[i] < running_steps[i]:
                    obs_i, episode_done = GDict(self.workers[i].wait()).slice(0, wrapper=False)
                    self.recent_obs.assign((i,), obs_i)
                    num_finished[i] += int(episode_done)
                    # Commenting this out for now; this causes pynvml.nvml.NVMLError_FunctionNotFound for some reason
                    # if i == 0 and bool(episode_done):
                    #     log_mem_info(self.logger)
        self.finish()
        if self.enable_merge:
            self.merge_results(n)
        return self.episode_lens, self.episode_rewards, self.episode_finishes

    def close(self):
        for worker in self.workers:
            worker.call("close")
            worker.close()
