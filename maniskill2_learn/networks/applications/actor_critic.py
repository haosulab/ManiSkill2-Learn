from gymnasium.spaces import Discrete, Box
from maniskill2_learn.utils.data import to_torch, GDict, DictArray, recover_with_mask
from maniskill2_learn.utils.torch import ExtendedModule, avg_grad
from ..builder import POLICYNETWORKS, VALUENETWORKS, build_backbone, build_reg_head

from ..utils import replace_placeholder_with_args, get_kwargs_from_shape, combine_obs_with_action
import torch, torch.nn as nn


class ActorCriticBase(ExtendedModule):

    def __init__(self, nn_cfg=None, head_cfg=None, mlp_cfg=None, backbone=None):
        super(ActorCriticBase, self).__init__()
        assert nn_cfg is None or backbone is None
        self.backbone = build_backbone(nn_cfg) if backbone is None else backbone
        self.final_mlp = build_backbone(mlp_cfg)
        self.head = build_reg_head(head_cfg)

    def forward(self, obs, actions=None, **kwargs):
        inputs = combine_obs_with_action(obs, actions)
        feature = self.backbone(inputs, **kwargs)
        if (
            isinstance(feature, dict) and "aux_loss" in feature.keys()
        ):  # auxiliary backbone self-supervision
            assert self.final_mlp is None, "when using auxiliary backbone self-supervision, returned feature should be the final feature"
            feature, aux_loss = feature["feat"], feature["aux_loss"]
        else:
            feature, aux_loss = feature, None
        kwargs.pop("feature", None)

        if self.final_mlp is not None:
            # MLP do not need any extra kwargs
            feature = self.final_mlp(feature)

        if self.head is not None:
            feature = self.head(feature, **kwargs)

        if aux_loss is None or not kwargs.pop("require_aux_loss", False):
            # return aux_loss only when the algorithm gd update, e.g. PPO update, requires it;
            # do NOT return aux_loss during e.g. rollout or evaluation
            return feature
        else:
            return {"feat": feature, "aux_loss": aux_loss}


@POLICYNETWORKS.register_module(name="ContinuousPolicy")
@POLICYNETWORKS.register_module()
class ContinuousActor(ActorCriticBase):
    def __init__(self, nn_cfg=None, head_cfg=None, mlp_cfg=None, backbone=None, action_space=None, obs_shape=None, action_shape=None, **kwargs):
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg, mlp_cfg = replace_placeholder_with_args([nn_cfg, mlp_cfg], **replaceable_kwargs)
        assert isinstance(action_space, Box), "If you are training over discrete action space, you need DiscreteActor"
        if head_cfg is not None and action_space is not None and action_space.is_bounded():
            head_cfg["bound"] = [action_space.low, action_space.high]
        super(ContinuousActor, self).__init__(nn_cfg=nn_cfg, head_cfg=head_cfg, mlp_cfg=mlp_cfg, backbone=backbone)


@POLICYNETWORKS.register_module()
class DiscreteActor(ActorCriticBase):
    def __init__(self, nn_cfg=None, head_cfg=None, mlp_cfg=None, backbone=None, action_space=None, obs_shape=None, action_shape=None, **kwargs):
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg, mlp_cfg = replace_placeholder_with_args([nn_cfg, mlp_cfg], **replaceable_kwargs)
        assert isinstance(action_space, Discrete), "If you are training over discrete action space, you need ContinuousActor"
        head_cfg["num_choices"] = action_shape
        super(DiscreteActor, self).__init__(nn_cfg=nn_cfg, head_cfg=head_cfg, mlp_cfg=mlp_cfg, backbone=backbone)


@POLICYNETWORKS.register_module(name="ContinuousValue")
@VALUENETWORKS.register_module()
class ContinuousCritic(ExtendedModule):
    # Or Value for discrete action space
    def __init__(
        self, nn_cfg=None, head_cfg=None, mlp_cfg=None, backbone=None, obs_shape=None, action_shape=None, num_heads=1, average_grad=True, **kwargs
    ):
        super(ContinuousCritic, self).__init__()
        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg, mlp_cfg = replace_placeholder_with_args([nn_cfg, mlp_cfg], **replaceable_kwargs)

        self.values = nn.ModuleList()
        self.num_heads = num_heads
        self.shared_feature = backbone is not None
        self.average_grad = average_grad
        if self.shared_feature and num_heads > 1:
            assert mlp_cfg is not None, "We should use different MLP in ContinuousCritic for shared multi-head critic!"

        for i in range(num_heads):
            self.values.append(ActorCriticBase(nn_cfg=nn_cfg, head_cfg=head_cfg, mlp_cfg=mlp_cfg, backbone=backbone))

    def forward(self, obs, actions=None, **kwargs):
        kwargs = dict(kwargs)

        if self.shared_feature:
            feature = self.values[0].backbone(obs=obs, actions=actions, **kwargs)
            if self.num_heads > 1:
                feature = avg_grad(feature, self.num_heads)
            ret = [value(obs=obs, actions=actions, feature=feature, **kwargs) for i, value in enumerate(self.values)]
        else:
            ret = [value(obs=obs, actions=actions, **kwargs) for i, value in enumerate(self.values)]
        return GDict.concat(ret, -1).contiguous(False)


@VALUENETWORKS.register_module()
class DiscreteCritic(ContinuousCritic):
    def forward(self, obs, actions=None, actions_prob=None, detach_value=False, **kwargs):
        assert not (actions is not None and actions_prob is not None), "We only need one of actions and actions_prob"
        kwargs = dict(kwargs)
        if self.shared_feature:
            feature = self.values[0].backbone(obs=obs, **kwargs)
            if self.num_heads > 1:
                feature = avg_grad(feature, self.num_heads)
            ret = [value(obs=obs, feature=feature, **kwargs) for i, value in enumerate(self.values)]
        else:
            ret = [value(obs=obs, **kwargs) for i, value in enumerate(self.values)]
        ret = GDict.stack(ret, -2).contiguous(False)  # [B, num_head, dim_value]

        if detach_value:
            ret = ret.detach()

        if actions_prob is not None:
            # Return V instead of Q when getting actions_prob
            actions_prob = actions_prob[..., None, :]
            ret = (ret * actions_prob).sum(-1)
        elif actions is not None:
            # print('Q', ret.shape)
            actions = torch.repeat_interleave(actions[..., None], ret.shape[-2], dim=-2)
            ret = torch.gather(ret, -1, actions)[..., 0]

        return ret