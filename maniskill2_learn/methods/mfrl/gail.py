"""
Generative Adversarial Imitation Learning
SAC version..
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from maniskill2_learn.networks import build_model, build_actor_critic
from maniskill2_learn.utils.torch import build_optimizer
from maniskill2_learn.utils.data import to_torch
from ..builder import MFRL
from maniskill2_learn.utils.torch import BaseAgent, hard_update, soft_update
from maniskill2_learn.utils.data import DictArray
import math

# Modified from Hao Shen, Weikang Wan, and He Wang's ManiSkill2021 challenge submission:
# Paper: https://arxiv.org/pdf/2203.02107.pdf
# Code: https://github.com/wkwan7/EPICLab-ManiSkill

@MFRL.register_module()
class GAIL(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        discriminator_cfg,
        env_params,
        batch_size=256,
        discriminator_batch_size=512,
        discriminator_update_freq=0.125,
        discriminator_update_n=5,
        episode_based_discriminator_update=True,
        env_reward_proportion=0.3,
        gamma=0.95,
        update_coeff=0.005,
        alpha=0.2,
        ignore_dones=False,
        target_update_interval=1,
        automatic_alpha_tuning=True,
        clip_reward=False,
        alpha_optim_cfg=None,
        target_entropy=None,
        use_demo_for_policy_update=False,
        shared_backbone=False,
        detach_actor_feature=False,
    ):
        super(GAIL, self).__init__()
        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)
        discriminator_cfg = deepcopy(discriminator_cfg)
        actor_optim_cfg = actor_cfg.pop("optim_cfg")
        critic_optim_cfg = critic_cfg.pop("optim_cfg")
        discriminator_optim_cfg = discriminator_cfg.pop("optim_cfg")
        action_shape = env_params["action_shape"]

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.ignore_dones = ignore_dones
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning

        actor_cfg.update(env_params)
        critic_cfg.update(env_params)
        discriminator_cfg.update(env_params)        

        self.discriminator_batch_size = discriminator_batch_size
        assert 0 < discriminator_update_freq and discriminator_update_freq <= 1
        self.discriminator_update_freq = discriminator_update_freq
        assert isinstance(discriminator_update_n, int) and discriminator_update_n >= 1
        self.discriminator_update_n = discriminator_update_n
        self.discriminator_counter = 0
        self.episode_based_discriminator_update = episode_based_discriminator_update

        self.env_reward_proportion = env_reward_proportion
        # Clip the modified reward to negative, to prevent agent being "stuck"
        self.clip_reward = clip_reward

        # during policy / value update, whether to sample from demo besides
        # sampling from replay buffer; however, empirically, if demo experiences
        # are trained for too long, then feature coadaptation issues could occur;
        # so generally a better way is to load demo experiences into replay buffer
        # (which will be replaced by online experiences when the buffer is full)
        self.use_demo_for_policy_update = use_demo_for_policy_update

        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)
        self.shared_backbone = shared_backbone
        self.detach_actor_feature = detach_actor_feature
        
        self.discriminator = build_model(discriminator_cfg)
        self.discriminator_criterion = nn.BCELoss()

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)
        self.discriminator_optim = build_optimizer(self.discriminator, discriminator_optim_cfg)
        
        self.target_critic = build_model(critic_cfg)
        hard_update(self.target_critic, self.critic)

        self.log_alpha = nn.Parameter(torch.ones(1, requires_grad=True) * np.log(alpha))
        if target_entropy is None:
            self.target_entropy = -np.prod(action_shape)
        else:
            self.target_entropy = target_entropy
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)

    def update_discriminator_helper(self, expert_replay, recent_traj_replay):
        expert_sampled_batch = expert_replay.sample(self.discriminator_batch_size // 2).to_torch(
            dtype="float32", device=self.device, non_blocking=True
        )
        recent_traj_sampled_batch = recent_traj_replay.sample(self.discriminator_batch_size // 2).to_torch(
            dtype="float32", device=self.device, non_blocking=True
        )
        expert_sampled_batch = self.process_obs(expert_sampled_batch)
        recent_traj_sampled_batch = self.process_obs(recent_traj_sampled_batch)

        expert_out = torch.sigmoid(self.discriminator(expert_sampled_batch["obs"], expert_sampled_batch["actions"]))
        recent_traj_out = torch.sigmoid(self.discriminator(recent_traj_sampled_batch["obs"], recent_traj_sampled_batch["actions"]))

        self.discriminator_optim.zero_grad()
        discriminator_loss = self.discriminator_criterion(
            expert_out, torch.zeros((expert_out.shape[0], 1), device=self.device)
        ) + self.discriminator_criterion(recent_traj_out, torch.ones((recent_traj_out.shape[0], 1), device=self.device))
        discriminator_loss = discriminator_loss.mean()
        discriminator_loss.backward()
        self.discriminator_optim.step()

    def update_discriminator(self, expert_replay, recent_traj_replay, n_finished_episodes):
        if self.episode_based_discriminator_update:
            self.discriminator_counter += n_finished_episodes
        else:
            self.discriminator_counter += 1
        if self.discriminator_counter >= math.ceil(1.0 / self.discriminator_update_freq):
            for _ in range(self.discriminator_update_n):
                self.update_discriminator_helper(expert_replay, recent_traj_replay)
            self.discriminator_counter = 0
            return True
        else:
            return False

    def update_parameters(self, memory, updates, expert_replay):
        if self.use_demo_for_policy_update:
            mem_sample_n = self.batch_size // 4 * 3
            demo_sample_n = self.batch_size - mem_sample_n
            sampled_batch = memory.sample(mem_sample_n).to_torch(dtype="float32", device=self.device, non_blocking=True)
            demo_sampled_batch = expert_replay.sample(demo_sample_n).to_torch(dtype="float32", device=self.device, non_blocking=True)
            sampled_batch = DictArray.concat([sampled_batch, demo_sampled_batch])
        else:
            sampled_batch = memory.sample(self.batch_size).to_torch(dtype="float32", device=self.device, non_blocking=True)

        sampled_batch = self.process_obs(sampled_batch)

        with torch.no_grad():
            discriminator_rewards = -torch.log(torch.sigmoid(self.discriminator(sampled_batch["obs"], sampled_batch["actions"])))
            assert sampled_batch["rewards"].size() == discriminator_rewards.size()
            old_rewards = sampled_batch["rewards"]
            sampled_batch["rewards"] = self.env_reward_proportion * old_rewards + (1 - self.env_reward_proportion) * discriminator_rewards
            if self.clip_reward:
                clip_max = torch.clamp(old_rewards, min=-0.5)
                sampled_batch["rewards"] = torch.clamp(sampled_batch["rewards"], max=clip_max)
            next_action, next_log_prob = self.actor(sampled_batch["next_obs"], mode="all")[:2]
            q_next_target = self.target_critic(sampled_batch["next_obs"], next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            if not self.ignore_dones:
                q_target = sampled_batch["rewards"] + (1 - sampled_batch["dones"]) * self.gamma * min_q_next_target
            else:
                q_target = sampled_batch["rewards"] + self.gamma * min_q_next_target

        q = self.critic(sampled_batch["obs"], sampled_batch["actions"])
        critic_loss = F.mse_loss(q, q_target.repeat(1, q.shape[-1])) * q.shape[-1]
        with torch.no_grad():
            abs_critic_error = torch.abs(q - q_target.repeat(1, q.shape[-1])).max().item()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        with torch.no_grad():
            critic_grad = self.critic.grad_norm
        if self.shared_backbone:
            self.critic_optim.zero_grad()        

        pi, log_pi = self.actor(sampled_batch["obs"], mode="all", 
            save_feature=self.shared_backbone, detach_visual=self.detach_actor_feature)[:2]
        entropy_term = -log_pi.mean()

        visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")
        if visual_feature is not None:
            visual_feature = visual_feature.detach()

        q_pi = self.critic(sampled_batch["obs"], pi, visual_feature=visual_feature)
        q_pi_min = torch.min(q_pi, dim=-1, keepdim=True).values
        actor_loss = -(q_pi_min - self.alpha * log_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        with torch.no_grad():
            actor_grad = self.actor.grad_norm

        if self.automatic_alpha_tuning:
            alpha_loss = self.log_alpha.exp() * (entropy_term - self.target_entropy).detach()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)

        return {
            "gail/discriminator_rewards": discriminator_rewards.mean().item(),
            "gail/critic_loss": critic_loss.item(),
            "gail/max_critic_abs_err": abs_critic_error,
            "gail/actor_loss": actor_loss.item(),
            "gail/alpha": self.alpha,
            "gail/alpha_loss": alpha_loss.item(),
            "gail/q": torch.min(q, dim=-1).values.mean().item(),
            "gail/q_target": torch.mean(q_target).item(),
            "gail/entropy": entropy_term.item(),
            "gail/target_entropy": self.target_entropy,
            "gail/critic_grad": critic_grad,
            "gail/actor_grad": actor_grad,
        }
