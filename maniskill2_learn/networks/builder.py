import torch.nn as nn

from ..utils.meta import Registry, build_from_cfg

BACKBONES = Registry("backbone")
APPLICATIONS = Registry("applications")
REGHEADS = Registry("regression_head")

POLICYNETWORKS = Registry("policy_network")
VALUENETWORKS = Registry("value_network")
MODELNETWORKS = Registry("model_network")


def build(cfg, registry, default_args=None):
    if cfg is None:
        return None
    elif isinstance(cfg, (list, tuple)):
        modules = [build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg]
        return modules  # nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_reg_head(cfg):
    return build(cfg, REGHEADS)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_model(cfg, default_args=None):
    if cfg is None:
        return None
    for model_type in [BACKBONES, POLICYNETWORKS, VALUENETWORKS, MODELNETWORKS]:
        if cfg["type"] in model_type.module_dict:
            return build(cfg, model_type, default_args)
    raise RuntimeError(f"This model type:{cfg['type']} does not exist!")


def build_actor_critic(actor_cfg, critic_cfg, shared_backbone=False):
    if shared_backbone:
        assert (
            actor_cfg["nn_cfg"]["type"] in ["Visuomotor", "FrameMiners"]
            or "Visuomotor" in actor_cfg["nn_cfg"]["type"]
        ), f"Only Visuomotor models can share visual backbone. Your model has type {actor_cfg['nn_cfg']['type']}!"
        actor = build_model(actor_cfg)

        if getattr(actor.backbone, "visual_nn", None) is not None:
            critic_cfg.nn_cfg.visual_nn_cfg = None
            critic_cfg.nn_cfg.visual_nn = actor.backbone.visual_nn

        critic = build_model(critic_cfg)
        return actor, critic
    else:
        return build_model(actor_cfg), build_model(critic_cfg)
