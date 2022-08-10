"""
Proximal Policy Optimization Algorithms (PPO):
    https://arxiv.org/pdf/1707.06347.pdf

Related Tricks(May not be useful):
    Mastering Complex Control in MOBA Games with Deep Reinforcement Learning (Dual Clip)
        https://arxiv.org/pdf/1912.09729.pdf
    A Closer Look at Deep Policy Gradients (Value clip, Reward normalizer)
        https://openreview.net/pdf?id=ryxdEkHtPS
    Revisiting Design Choices in Proximal Policy Optimization
        https://arxiv.org/pdf/2009.10897.pdf

Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (DAPG):
        https://arxiv.org/pdf/1709.10087.pdf
"""

from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from maniskill2_learn.env import build_replay
from maniskill2_learn.networks import build_actor_critic, build_model
from maniskill2_learn.utils.torch import build_optimizer
from maniskill2_learn.utils.data import DictArray, GDict, to_np, to_torch
from maniskill2_learn.utils.meta import get_logger, get_world_rank, get_world_size
from maniskill2_learn.utils.torch import BaseAgent, RunningMeanStdTorch, RunningSecondMomentumTorch, barrier, get_flat_grads, get_flat_params, set_flat_grads

from ..builder import MFRL


@MFRL.register_module()
class PPO(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        env_params,

        gamma=0.99,
        lmbda=0.95,
        max_kl=None,

        obs_norm=False,
        rew_norm=True,
        adv_norm=True,
        recompute_value=True,

        eps_clip=0.2,
        critic_coeff=0.5,
        entropy_coeff=0.0,
        num_epoch=10,
        critic_epoch=-1,
        actor_epoch=-1,
        num_mini_batch=-1,
        critic_warmup_epoch=0,
        batch_size=256,

        max_grad_norm=0.5,
        rms_grad_clip=None,
        
        dual_clip=None,
        critic_clip=False,
        
        shared_backbone=False,
        detach_actor_feature=True,
        debug_grad=False,
        
        demo_replay_cfg=None,
        dapg_lambda=0.1,
        dapg_damping=0.995,
        ignore_dones=True,
        visual_state_coeff=-1,
        visual_state_mlp_cfg=None,
        **kwargs
    ):
        super(PPO, self).__init__()
        assert dual_clip is None or dual_clip > 1.0, "Dual-clip PPO parameter should greater than 1.0."
        assert max_grad_norm is None or rms_grad_clip is None, "Only one gradient clip mode is allowed!"
        assert (
            (num_epoch > 0 and (actor_epoch < 0 and critic_epoch < 0)) or (num_epoch < 0 and (actor_epoch > 0 and critic_epoch > 0)),
            "We need only one set of the parameters num_epoch > 0, (actor_epoch > 0 and critic_epoch > 0).",
        )
        if not rew_norm:
            assert not critic_clip, "Value clip is available only when `reward_normalization` is True"

        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)

        actor_optim_cfg = actor_cfg.pop("optim_cfg", None)
        critic_optim_cfg = critic_cfg.pop("optim_cfg", None)
        obs_shape = env_params["obs_shape"]
        self.is_discrete = env_params["is_discrete"]

        self.gamma = gamma
        self.lmbda = lmbda
        
        self.adv_norm = adv_norm
        self.obs_rms = RunningMeanStdTorch(obs_shape, clip_max=10) if obs_norm else None
        self.rew_rms = RunningMeanStdTorch(1) if rew_norm else None

        self.critic_coeff = critic_coeff
        self.entropy_coeff = entropy_coeff
        self.eps_clip = eps_clip
        self.dual_clip = dual_clip
        self.critic_clip = critic_clip
        self.max_kl = max_kl
        self.recompute_value = recompute_value
        self.max_grad_norm = max_grad_norm
        self.rms_grad_clip = rms_grad_clip
        
        self.debug_grad = debug_grad

        self.num_mini_batch = num_mini_batch
        self.batch_size = batch_size  # The batch size for policy gradient

        self.critic_warmup_epoch = critic_warmup_epoch
        self.num_epoch = num_epoch
        self.critic_epoch = critic_epoch
        self.actor_epoch = actor_epoch

        # Use extra state to get better feature
        self.regress_visual_state = visual_state_coeff > 0 and visual_state_mlp_cfg is not None and "visual_state" in obs_shape
        self.visual_state_coeff = visual_state_coeff
        if self.regress_visual_state:
            assert shared_backbone, "Only Visuomotor policy supports extra state fitting"

        # For DAPG
        self.dapg_lambda = nn.Parameter(to_torch(dapg_lambda), requires_grad=False)
        self.dapg_damping = dapg_damping
        self.demo_replay = build_replay(demo_replay_cfg)
        if self.demo_replay is not None:
            for key in ['obs', 'actions']:
                assert key in self.demo_replay.memory, f"DAPG needs {key} in your demo!"

        # For done signal process.
        self.ignore_dones = ignore_dones

        # Build networks
        actor_cfg.update(env_params)
        critic_cfg.update(env_params)

        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)

        if self.regress_visual_state:
            visual_state_mlp_cfg.mlp_spec += [obs_shape["visual_state"]]
            self.extra_fit = build_model(visual_state_mlp_cfg)

        if rms_grad_clip is not None:
            self.grad_rms = RunningSecondMomentumTorch(get_flat_params(self, trainable=True).shape, clip_max=rms_grad_clip)

        self.shared_backbone = shared_backbone
        self.detach_actor_feature = detach_actor_feature

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)

    def compute_critic_loss(self, samples):
        # For update_actor_critic and update critic
        assert isinstance(samples, (dict, GDict))
        values = self.critic(
            samples["obs"], episode_dones=samples["episode_dones"], save_feature=True
        )
        feature = self.critic.values[0].backbone.pop_attr("saved_feature")
        visual_feature = self.critic.values[0].backbone.pop_attr("saved_visual_feature")
        if self.detach_actor_feature and feature is not None:
            feature = feature.detach()

        if self.critic_clip and isinstance(self.critic_clip, float):
            v_clip = samples["old_values"] + (values - samples["old_values"]).clamp(-self.critic_clip, self.critic_clip)
            vf1 = (samples["returns"] - values).pow(2)
            vf2 = (samples["returns"] - v_clip).pow(2)
            critic_loss = torch.max(vf1, vf2)
        else:
            critic_loss = (samples["returns"] - values).pow(2)

        critic_loss = critic_loss.mean() if samples["is_valid"] is None else critic_loss[samples["is_valid"]].mean()
        return critic_loss, feature, visual_feature

    def update_actor_critic(self, samples, demo_samples=None, with_critic=False):
        """
        Returns True if self.max_kl is not None and
        policy update causes large kl divergence between new policy and old policy,
        in which case we stop the policy update and throw away the current replay buffer
        """
        is_valid = samples["is_valid"]
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        ret = {}
        critic_loss, actor_loss, demo_actor_loss, visual_state_loss, entropy_term = [0.0] * 5
        feature, visual_feature, critic_loss, policy_std = [None] * 4
        
        if with_critic:
            critic_mse, feature, visual_feature = self.compute_critic_loss(samples)
            critic_loss = critic_mse * self.critic_coeff
            ret["ppo/critic_err"] = critic_mse.item()
            # ret['ppo/critic_loss'] = critic_loss.item()

        # Run actor forward
        alls = self.actor(
            samples["obs"],
            episode_dones=samples["episode_dones"],
            mode="dist" if self.is_discrete else "dist_std",
            feature=feature,
            save_feature=feature is None,
            require_aux_loss=True, # auxiliary backbone self-supervision, e.g. aux_regress in VisuomotorTransformerFrame
        )
        if isinstance(alls, dict) and 'aux_loss' in alls.keys(): # auxiliary backbone self-supervision, e.g. aux_regress in VisuomotorTransformerFrame
            alls, backbone_aux_loss = alls['feat'], alls['aux_loss']
        else:
            backbone_aux_loss = None

        if not self.is_discrete:
            new_distributions, policy_std = alls
        else:
            new_distributions, policy_std = alls, None
        del alls

        if visual_feature is None:
            visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")

        # Compute actor loss
        dist_entropy = new_distributions.entropy().mean()
        recent_log_p = new_distributions.log_prob(samples["actions"])
        log_ratio = recent_log_p - samples["old_log_p"]
        ratio = log_ratio.exp()
        # print("ratio", ratio[:20], flush=True)

        # Estimation of KL divergence = p (log p - log q) with method in Schulman blog: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl_div = (ratio - 1 - log_ratio).mean().item()
            clip_frac = (torch.abs(ratio - 1) > self.eps_clip).float().mean().item()
            if policy_std is not None:
                ret["ppo/policy_std"] = policy_std.mean().item()
            ret["ppo/entropy"] = dist_entropy.item()
            ret["ppo/mean_p_ratio"] = ratio.mean().item()
            ret["ppo/max_p_ratio"] = ratio.max().item()
            ret["ppo/log_p"] = recent_log_p.mean().item()
            ret["ppo/clip_frac"] = clip_frac
            ret["ppo/approx_kl"] = approx_kl_div

        sign = GDict(self.max_kl is not None and approx_kl_div > self.max_kl * 1.5).allreduce(op="BOR", wrapper=False)

        if sign:
            return True, ret

        if ratio.ndim == samples["advantages"].ndim - 1:
            ratio = ratio[..., None]

        surr1 = ratio * samples["advantages"]
        surr2 = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * samples["advantages"]
        surr = torch.min(surr1, surr2)
        if self.dual_clip:
            surr = torch.max(surr, self.dual_clip * samples["advantages"])
        actor_loss = -surr[is_valid].mean()
        entropy_term = -dist_entropy * self.entropy_coeff
        ret["ppo/actor_loss"] = actor_loss.item()
        ret["ppo/entropy_loss"] = entropy_term.item()

        # DAPG actor loss
        if demo_samples is not None:
            new_demo_distributions = self.actor(demo_samples["obs"], mode="dist")
            nll_loss_demo = -new_demo_distributions.log_prob(demo_samples["actions"]).mean()
            demo_actor_loss = nll_loss_demo * self.dapg_lambda
            with torch.no_grad():
                ret["dapg/demo_nll_loss"] = nll_loss_demo.item()
                ret["dapg/demo_actor_loss"] = demo_actor_loss.item()

        # State regression loss
        if self.regress_visual_state:
            assert feature is not None
            visual_state_mse = F.mse_loss(self.extra_fit(visual_feature), samples["obs/visual_state"], reduction="none")
            visual_state_mse = visual_state_mse[is_valid].mean()
            ret["ppo-extra/visual_state_mse"] = visual_state_mse
            visual_state_loss = visual_state_mse * self.visual_state_coeff
            ret["ppo-extra/visual_state_loss"] = visual_state_loss.item()

        # Backbone auxiliary supervision loss
        if backbone_aux_loss is not None:
            ret["ppo-extra/backbone_auxiliary_loss"] = backbone_aux_loss.item()

        loss = actor_loss + entropy_term + critic_loss + visual_state_loss + demo_actor_loss
        if backbone_aux_loss is not None:
            loss = loss + backbone_aux_loss
        loss.backward()

        net = self if with_critic else self.actor
        ret["grad/grad_norm"] = net.grad_norm
        if math.isnan(ret["grad/grad_norm"]):
            print("############ Debugging nan grad ############", flush=True)
            print("Dist mean", new_distributions.mean, flush=True)
            print("Dist std", new_distributions.stddev, flush=True)
            print("Samples[actions]", samples["actions"], flush=True)
            print("Recent_log_p", recent_log_p, flush=True)
            print("Samples[old_log_p]", samples["old_log_p"], flush=True)
            for k, v in ret.keys():
                print(k, v, flush=True)

        if self.shared_backbone:
            if getattr(self.actor.backbone, "visual_nn", None) is not None:
                ret["grad/visual_grad"] = self.actor.backbone.visual_nn.grad_norm

            if getattr(self.actor.backbone, "final_mlp", None) is not None:
                ret["grad/actor_mlp_grad"] = self.actor.backbone.final_mlp.grad_norm
            elif self.actor.final_mlp is not None:
                ret["grad/actor_mlp_grad"] = self.actor.final_mlp.grad_norm

            if with_critic:
                if getattr(self.critic.values[0].backbone, "final_mlp", None) is not None:
                    ret["grad/critic_mlp_grad"] = self.critic.values[0].backbone.final_mlp.grad_norm
                elif self.critic.values[0].final_mlp is not None:
                    ret["grad/critic_mlp_grad"] = self.critic.values[0].final_mlp.grad_norm

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
        elif self.rms_grad_clip is not None:
            grads = get_flat_grads(self)
            grads = self.grad_rms.add(grads)
            set_flat_grads(self, grads)
        ret["grad/clipped_grad_norm"] = net.grad_norm

        self.actor_optim.step()
        if with_critic:
            self.critic_optim.step()

        return False, ret

    def update_critic(self, samples, demo_samples=None):
        self.critic_optim.zero_grad()
        critic_mse = self.compute_critic_loss(samples)[0]
        critic_loss = critic_mse * self.critic_coeff
        critic_loss.backward()

        ret = {}
        ret["grad/grad_norm"] = self.critic.grad_norm
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        elif self.rms_grad_clip is not None:
            assert False
            grads = get_flat_grads(self)
            grads = self.grad_rms.add(grads)
            set_flat_grads(self, grads)

        ret["grad/clipped_grad_norm"] = self.critic.grad_norm
        ret["ppo/critic_loss"] = critic_loss.item()
        ret["ppo/critic_mse"] = critic_mse.item()
        self.critic_optim.step()
        return ret

    def update_parameters(self, memory, updates, with_v=False):
        world_size = get_world_size()
        logger = get_logger()
        ret = defaultdict(list)

        process_batch_size = self.batch_size if GDict(memory["obs"]).is_big else None
        if self.num_mini_batch < 0:
            max_samples = GDict(len(memory)).allreduce(op="MAX", device=self.device, wrapper=False) if world_size > 1 else len(memory)
            num_mini_batch = int((max_samples + self.batch_size - 1) // self.batch_size)
        else:
            num_mini_batch = self.num_mini_batch
        logger.info(f"Number of batches in one PPO epoch: {num_mini_batch}!")

        if len(memory) < memory.capacity:
            memory["episode_dones"][len(memory) :] = True

        # Do transformation for all valid samples
        memory["episode_dones"] = (memory["episode_dones"] + memory["is_truncated"]) > 1 - 0.1
        if self.has_obs_process:
            self.obs_rms.sync()
            obs = GDict({"obs": memory["obs"], "next_obs": memory["next_obs"]}).to_torch(device="cpu", wrapper=False)
            obs = GDict(self.process_obs(obs, batch_size=process_batch_size)).to_numpy(wrapper=False)
            memory.update(obs)

        with torch.no_grad():
            memory["old_distribution"], memory["old_log_p"] = self.get_dist_with_logp(
                obs=memory["obs"], actions=memory["actions"], batch_size=process_batch_size
            )
            ret["ppo/old_log_p"].append(memory["old_log_p"].mean().item())

        demo_memory = self.demo_replay
        if demo_memory is not None:
            with torch.no_grad():
                demo_memory = self.demo_replay.sample(min(len(self.demo_replay), len(memory)))
                if self.has_obs_process:
                    demo_memory = demo_memory.to_torch(device="cpu")
                    demo_memory = self.process_obs(demo_memory, batch_size=process_batch_size)
                    demo_memory = demo_memory.to_numpy()
                if self.ignore_dones:
                    demo_memory["dones"] = demo_memory["dones"] * 0

        def run_over_buffer(epoch_id, mode="v"):
            nonlocal memory, ret, demo_memory, logger
            assert mode in ["v", "pi", "v+pi"]

            if "v" in mode and (epoch_id == 0 or self.recompute_value):
                with self.critic.no_sync():
                    memory.update(
                        self.compute_gae(
                            obs=memory["obs"],
                            next_obs=memory["next_obs"],
                            rewards=memory["rewards"],
                            dones=memory["dones"],
                            episode_dones=memory["episode_dones"],
                            update_rms=True,
                            batch_size=process_batch_size,
                            ignore_dones=self.ignore_dones,
                        )
                    )

                if self.adv_norm:
                    # print(mean_adv, std_adv)
                    mean_adv = memory["advantages"].mean(0)
                    std_adv = memory["advantages"].std(0) + 1e-8
                    mean_adv, std_adv = GDict([mean_adv, std_adv]).allreduce(wrapper=False)
                    # print(mean_adv, std_adv)
                    # exit(0)
                    memory["advantages"] = (memory["advantages"] - mean_adv) / std_adv
                    ret["ppo/adv_mean"].append(mean_adv.item())
                    ret["ppo/adv_std"].append(std_adv.item())
                    ret["ppo/max_normed_adv"].append(np.abs(memory["advantages"]).max().item())

                ret["ppo/v_target"].append(memory["returns"].mean().item())
                ret["ppo/ori_returns"].append(memory["original_returns"].mean().item())

            def run_one_iter(samples, demo_samples):
                if "pi" in mode:
                    flag, infos = self.update_actor_critic(samples, demo_samples, with_critic=(mode == "v+pi"))
                    for key in infos:
                        ret[key].append(infos[key])
                elif mode == "v":
                    flag, infos = False, self.update_critic(samples, demo_samples)
                    for key in infos:
                        ret[key].append(infos[key])
                return flag

            for samples in memory.mini_batch_sampler(self.batch_size, drop_last=True, auto_restart=True, max_num_batches=num_mini_batch):
                samples = DictArray(samples).to_torch(device=self.device, non_blocking=True)
                demo_samples = None
                if demo_memory is not None:
                    indices = np.random.randint(0, high=len(demo_memory), size=self.batch_size)
                    demo_samples = demo_memory.slice(indices).to_torch(device=self.device, non_blocking=True)
                if run_one_iter(samples, demo_samples):
                    return True

            return False

        if self.critic_warmup_epoch > 0:
            logger.info("**Warming up critic at the beginning of training; this causes reported ETA to be slower than actual ETA**")
        for i in range(self.critic_warmup_epoch):
            run_over_buffer(i, "v")

        if self.num_epoch > 0:
            for i in range(self.num_epoch):
                num_actor_epoch = i + 1
                if run_over_buffer(i, "v+pi"):
                    break
        else:
            for i in range(self.critic_epoch):
                run_over_buffer(i, "v")
            for i in range(self.actor_epoch):
                num_actor_epoch = i + 1
                if run_over_buffer(i, "pi"):
                    break
        self.critic_warmup_epoch = 0
        ret = {key: np.mean(ret[key]) for key in ret}
        with torch.no_grad():
            ret["param/max_policy_abs"] = torch.max(torch.abs(get_flat_params(self.actor))).item()
            ret["param/policy_norm"] = torch.norm(get_flat_params(self.actor)).item()
            if isinstance(self.critic, nn.Module):
                ret["param/max_critic_abs"] = torch.max(torch.abs(get_flat_params(self.critic))).item()
                ret["param/critic_norm"] = torch.norm(get_flat_params(self.critic)).item()

        for key in ["old_distribution", "old_log_p", "old_values", "old_next_values", "original_returns", "returns", "advantages"]:
            if key in memory.memory:
                memory.memory.pop(key)

        ret["ppo/num_actor_epoch"] = num_actor_epoch
        if self.demo_replay is not None:
            # For DAPG
            ret["dapg/demo_lambda"] = self.dapg_lambda.item()
            self.dapg_lambda *= self.dapg_damping
        if with_v:
            # For PPG
            ret["vf"] = to_np(memory["original_returns"])
        # exit(0)
        return ret
