"""
Soft Actor-Critic Algorithms and Applications:
    https://arxiv.org/abs/1812.05905
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor:
   https://arxiv.org/abs/1801.01290
"""
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from maniskill2_learn.networks import build_model, build_actor_critic
from maniskill2_learn.utils.torch import build_optimizer
from maniskill2_learn.utils.torch import BaseAgent, hard_update, soft_update
from ..builder import MFRL


@MFRL.register_module()
class SAC(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        env_params,
        batch_size=128,
        gamma=0.99,
        update_coeff=0.005,
        alpha=0.2,
        ignore_dones=True,
        target_update_interval=1,
        automatic_alpha_tuning=True,
        target_smooth=0.90,  # For discrete SAC
        alpha_optim_cfg=None,
        target_entropy=None,
        shared_backbone=False,
        detach_actor_feature=False,
    ):
        super(SAC, self).__init__()
        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)

        actor_optim_cfg = actor_cfg.pop("optim_cfg")
        critic_optim_cfg = critic_cfg.pop("optim_cfg")
        action_shape = env_params["action_shape"]
        self.is_discrete = env_params["is_discrete"]

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.ignore_dones = ignore_dones
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning

        actor_cfg.update(env_params)
        critic_cfg.update(env_params)

        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)
        self.shared_backbone = shared_backbone
        self.detach_actor_feature = detach_actor_feature

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)

        self.target_critic = build_model(critic_cfg)
        hard_update(self.target_critic, self.critic)

        self.log_alpha = nn.Parameter(torch.ones(1, requires_grad=True) * np.log(alpha))
        if target_entropy is None:
            if env_params["is_discrete"]:
                # Use label smoothing to get the target entropy.
                n = np.prod(action_shape)
                explore_rate = (1 - target_smooth) / (n - 1)
                self.target_entropy = -(target_smooth * np.log(target_smooth) + (n - 1) * explore_rate * np.log(explore_rate))
                self.log_alpha = nn.Parameter(torch.tensor(np.log(0.1), requires_grad=True))
                # self.target_entropy = np.log(action_shape) * target_smooth
            else:
                self.target_entropy = -np.prod(action_shape)
        else:
            self.target_entropy = target_entropy
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)

    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size).to_torch(device=self.device, non_blocking=True)
        sampled_batch = self.process_obs(sampled_batch)
        with torch.no_grad():
            if self.is_discrete:
                _, next_action_prob, _, _, dist_next = self.actor(sampled_batch["next_obs"], mode="all_discrete")
                q_next_target = self.target_critic(sampled_batch["next_obs"], actions_prob=next_action_prob)
                min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values + self.alpha * dist_next.entropy()[..., None]
            else:
                next_action, next_log_prob = self.actor(sampled_batch["next_obs"], mode="all")[:2]
                q_next_target = self.target_critic(sampled_batch["next_obs"], next_action)
                min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            if self.ignore_dones:
                q_target = sampled_batch["rewards"] + self.gamma * min_q_next_target
            else:
                q_target = sampled_batch["rewards"] + (1 - sampled_batch["dones"].float()) * self.gamma * min_q_next_target
            q_target = q_target.repeat(1, q_next_target.shape[-1])
        q = self.critic(sampled_batch["obs"], sampled_batch["actions"])

        critic_loss = F.mse_loss(q, q_target) * q.shape[-1]
        with torch.no_grad():
            abs_critic_error = torch.abs(q - q_target).max().item()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        with torch.no_grad():
            critic_grad = self.critic.grad_norm
        if self.shared_backbone:
            self.critic_optim.zero_grad()

        if self.is_discrete:
            _, pi, _, _, dist = self.actor(sampled_batch["obs"], mode="all_discrete", save_feature=self.shared_backbone, detach_visual=self.detach_actor_feature)
            entropy_term = dist.entropy().mean()
        else:
            pi, log_pi = self.actor(sampled_batch["obs"], mode="all", save_feature=self.shared_backbone, detach_visual=self.detach_actor_feature)[:2]
            entropy_term = -log_pi.mean()

        visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")
        if visual_feature is not None:
            visual_feature = visual_feature.detach()

        if self.is_discrete:
            q = self.critic(sampled_batch["obs"], visual_feature=visual_feature, detach_value=True).min(-2).values
            q_pi = (q * pi).sum(-1)
            with torch.no_grad():
                q_match_rate = (pi.argmax(-1) == q.argmax(-1)).float().mean().item()
        else:
            q_pi = self.critic(sampled_batch["obs"], pi, visual_feature=visual_feature)
            q_pi = torch.min(q_pi, dim=-1, keepdim=True).values
        actor_loss = -(q_pi.mean() + self.alpha * entropy_term)
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

        ret = {
            "sac/critic_loss": critic_loss.item(),
            "sac/max_critic_abs_err": abs_critic_error,
            "sac/actor_loss": actor_loss.item(),
            "sac/alpha": self.alpha,
            "sac/alpha_loss": alpha_loss.item(),
            "sac/q": torch.min(q, dim=-1).values.mean().item(),
            "sac/q_target": torch.mean(q_target).item(),
            "sac/entropy": entropy_term.item(),
            "sac/target_entropy": self.target_entropy,
            "sac/critic_grad": critic_grad,
            "sac/actor_grad": actor_grad,
        }
        if self.is_discrete:
            ret["sac/q_match_rate"] = q_match_rate

        return ret
