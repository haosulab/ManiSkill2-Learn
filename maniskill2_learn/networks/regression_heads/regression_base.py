import torch, torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

from maniskill2_learn.utils.data import is_num, is_not_null, to_np
from maniskill2_learn.utils.torch import ExtendedModule, CustomCategorical
from ..builder import build_backbone, REGHEADS


class ContinuousBaseHead(ExtendedModule):
    def __init__(self, bound=None, dim_output=None, nn_cfg=None, clip_return=False, num_heads=1):
        super(ContinuousBaseHead, self).__init__()
        self.bound = bound
        self.net = build_backbone(nn_cfg)
        self.clip_return = clip_return and is_not_null(bound)

        if is_not_null(bound):
            if is_num(bound[0]):
                bound[0] = np.ones(dim_output) * bound[0]
            if is_num(bound[1]):
                bound[1] = np.ones(dim_output) * bound[1]
            assert (to_np(bound[0].shape) == to_np(bound[1].shape)).all()
            assert dim_output is None or bound[0].shape[-1] == dim_output
            dim_output = bound[0].shape[-1]
            if bound[0].ndim > 1:
                assert bound[0].ndim == 2 and bound[0].shape[0] == num_heads and num_heads > 1
            self.lb, self.ub = [Parameter(torch.tensor(bound[i]), requires_grad=False) for i in [0, 1]]
            self.log_uniform_prob = torch.log(1.0 / ((self.ub - self.lb).data)).sum().item()
            self.scale = Parameter(torch.tensor(bound[1] - bound[0]) / 2, requires_grad=False)
            self.bias = Parameter(torch.tensor(bound[0] + bound[1]) / 2, requires_grad=False)
        else:
            self.scale, self.bias = 1, 0

        self.dim_output = dim_output
        self.num_heads = num_heads
        self.dim_feature = None

    def uniform(self, sample_shape):
        r = torch.rand(sample_shape, self.dim_output, device=self.device)
        return (r * self.ub + (1 - r) * self.lb), torch.ones(sample_shape, device=self.device) * self.log_uniform_prob

    def clamp(self, x):
        if self.clip_return:
            x = torch.clamp(x, min=self.lb, max=self.ub)
        return x

    def forward(self, feature, **kwargs):
        return self.net(feature) if self.net is not None else feature


@REGHEADS.register_module()
class DiscreteBaseHead(ExtendedModule):
    def __init__(self, num_choices, num_heads=1, **kwargs):
        super(DiscreteBaseHead, self).__init__()
        assert num_heads == 1, "We only support one head recently."
        self.num_choices = num_choices
        self.num_heads = num_heads

    def forward(self, feature, num_actions=1, mode="explore", **kwargs):
        assert feature.shape[-1] == self.num_choices * self.num_heads
        feature = feature.repeat_interleave(num_actions, dim=0)
        dist = CustomCategorical(logits=feature)

        if "std" in mode or mode == "all":
            raise ValueError("Discrete action do not support log std..")

        if mode == "mean" or mode == "eval":
            return feature.argmax(dim=-1, keepdim=True)
        elif mode == "dist":
            return dist
        elif mode == "dist_mean":
            return dist, feature.argmax(dim=-1, keepdim=True)
        elif mode == "explore" or mode == "sample":
            sample = dist.rsample() if dist.has_rsample else dist.sample()
            return sample[..., None]
        elif mode == "all_discrete":
            # For SAC only, num_heads > 1 are not supported recently..

            # sample = dist.rsample() if dist.has_rsample else dist.sample( )
            # log_p = dist.log_prob(sample)
            # sample, log_p = dist.rsample_with_log_prob()
            # log_p = log_p[..., None]
            return feature, dist.probs, feature.argmax(dim=-1, keepdim=True), None, dist
        else:
            raise ValueError(f"Unsupported mode {mode}!!")

    def extra_repr(self) -> str:
        return f"num_actions={self.num_choices}, num_head={self.num_heads}"
