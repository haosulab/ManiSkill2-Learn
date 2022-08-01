"""
Behavior cloning(BC)
"""
from random import sample
from itertools import chain
from tqdm import tqdm
import numpy as np
from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from maniskill2_learn.networks import build_model, ContinuousActor, build_reg_head
from maniskill2_learn.schedulers import build_lr_scheduler

from maniskill2_learn.utils.data import to_torch, DictArray, GDict, dict_to_str
from maniskill2_learn.utils.meta import get_total_memory, get_logger
from maniskill2_learn.utils.torch import BaseAgent, get_mean_lr, get_cuda_info, build_optimizer

from ..builder import BRL


@BRL.register_module()
class BC(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        env_params,
        batch_size=128,
        loss_type="mse_loss",
        max_grad_norm=None,
        **kwargs,
    ):
        super(BC, self).__init__()
        self.batch_size = batch_size

        actor_optim_cfg = actor_cfg.pop("optim_cfg")
        lr_scheduler_cfg = actor_cfg.pop("lr_scheduler_cfg", None)
        actor_cfg.update(env_params)
        action_shape = env_params['action_shape']

        self.action_size = np.prod(action_shape)
        self.actor = build_model(actor_cfg)

        self.max_grad_norm = max_grad_norm
        assert isinstance(self.actor, ContinuousActor)

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        if lr_scheduler_cfg is None:
            self.lr_scheduler = None
        else:
            lr_scheduler_cfg["optimizer"] = self.actor_optim
            self.lr_scheduler = build_lr_scheduler(lr_scheduler_cfg)

        self.loss_type = loss_type

        self.extra_parameters = dict(kwargs)

    def forward(self, obs, **kwargs):
        obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
        return self.actor(obs, **kwargs)

    def compute_regression_loss(self, pred_dist, pred_action, target_action, mask=None):
        if mask is None:
            mask = torch.ones_like(target_action[..., 0])
        assert mask.ndim == target_action.ndim - 1
        print_dict = defaultdict(list)
        print_dict["abs_err"] = ((torch.abs(pred_action - target_action).mean(-1) * mask).sum() / mask.sum()).item()
        if hasattr(F, self.loss_type):
            assert self.loss_type in ["mse_loss", "l1_loss", "smooth_l1_loss"]
            if self.actor.head.num_heads == 1:
                actor_loss = (getattr(F, self.loss_type)(pred_action, target_action, reduction="none").mean(-1) * mask).sum() / mask.sum()
            else:
                # Mixture of expert
                log_mix_prob = torch.log_softmax(pred_dist.mixture_distribution.logits, dim=-1)
                all_mean = pred_dist.component_distribution.mean
                error = (getattr(F, self.loss_type)(all_mean, target_action[..., None, :], reduction="none").mean(-1)) * self.extra_parameters.get(
                    "moe_error_coeff", 1
                )
                actor_loss = -(torch.logsumexp(log_mix_prob - error, dim=-1) * mask).sum() / mask.sum()
            print_dict[f"{self.loss_type}"] = actor_loss.item()
        elif self.loss_type == "nll":
            actor_loss = -(pred_dist.log_prob(target_action) / self.action_size * mask).sum() / mask.sum()
            print_dict["nll_loss"] = actor_loss.item()
        elif self.loss_type == "cluster":
            cluster_target = self.actor.head.project_target(target_action)
            actor_loss = F.cross_entropy(
                pred_dist, cluster_target
            )
            print_dict["cluster_loss"] = actor_loss.item()

        return actor_loss, print_dict

    def update_parameters(self, memory, updates):
        batch_size = self.batch_size
        sampled_batch = memory.sample(batch_size).to_torch(dtype="float32")

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        sampled_batch = sampled_batch.to_torch(device=self.device, dtype="float32", non_blocking=True)
        self.actor_optim.zero_grad()
        [pred_dist, pred_action] = self.actor(sampled_batch["obs"], mode="dist_mean")
        loss, ret_dict = self.compute_regression_loss(pred_dist, pred_action, sampled_batch["actions"])
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(chain(self.actor.parameters(), self.critic.parameters()), self.max_grad_norm)
        self.actor_optim.step()

        ret_dict["grad_norm"] = np.mean(
            [torch.linalg.norm(parameter.grad.data).item() for parameter in self.actor.parameters() if parameter.grad is not None]
        )

        if self.lr_scheduler is not None:
            ret_dict["lr"] = get_mean_lr(self.actor_optim)
        ret_dict = dict(ret_dict)
        ret_dict = {'bc/' + key: val for key, val in ret_dict.items()}
        return ret_dict

    def compute_test_loss(self, memory):
        logger = get_logger()
        logger.info(f"Begin to compute test loss with batch size {self.batch_size}!")
        ret_dict = {}
        num_samples = 0

        from maniskill2_learn.utils.meta import TqdmToLogger
        from tqdm import tqdm

        tqdm_obj = tqdm(total=memory.data_size, file=TqdmToLogger(), mininterval=20)

        batch_size = self.batch_size         
        for sampled_batch in memory.mini_batch_sampler(self.batch_size, drop_last=False):
            sampled_batch = sampled_batch.to_torch(device="cuda", dtype="float32", non_blocking=True)

            is_valid = sampled_batch["is_valid"].squeeze(-1)
            pred_dist, pred_action = self.actor(sampled_batch["obs"], mode="dist_mean")
            loss, print_dict = self.compute_regression_loss(pred_dist, pred_action, sampled_batch["actions"])
            for key in print_dict:
                ret_dict[key] = ret_dict.get(key, 0) + print_dict[key] * len(sampled_batch)
            num_samples += len(sampled_batch)
            tqdm_obj.update(len(sampled_batch))

        logger.info(f"We compute the test loss over {num_samples} samples!")

        print_dict = {}
        print_dict["memory"] = get_total_memory("G", False)
        print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))
        print_info = dict_to_str(print_dict)
        logger.info(print_info)

        for key in ret_dict:
            ret_dict[key] /= num_samples

        return ret_dict
