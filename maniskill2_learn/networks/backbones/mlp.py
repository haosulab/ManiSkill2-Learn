from maniskill2_learn.networks.modules.norm import need_bias
import torch.nn as nn, torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from einops.layers.torch import Rearrange

from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.torch import load_checkpoint, ExtendedModule

from ..builder import BACKBONES
from ..modules import ConvModule, build_init, MLP, SharedMLP
from ..modules import build_activation_layer, build_norm_layer

BACKBONES.register_module(name="MLP", module=MLP)
BACKBONES.register_module(name="SharedMLP", module=SharedMLP)


@BACKBONES.register_module()
class LinearMLP(ExtendedModule):
    def __init__(
        self,
        mlp_spec,
        norm_cfg=dict(type="LN1d"), # Change BN -> LN
        bias="auto",
        act_cfg=dict(type="ReLU"),
        inactivated_output=True,
        zero_init_output=False,
        pretrained=None,
        linear_init_cfg=None,
        norm_init_cfg=None,
    ):
        super(LinearMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            bias_i = need_bias(norm_cfg) if bias == "auto" else bias
            self.mlp.add_module(f"linear{i}", nn.Linear(mlp_spec[i], mlp_spec[i + 1], bias=bias_i))
            if norm_cfg:
                self.mlp.add_module(f"norm{i}", build_norm_layer(norm_cfg, mlp_spec[i + 1])[1])
            if act_cfg:
                self.mlp.add_module(f"act{i}", build_activation_layer(act_cfg))
        self.init_weights(pretrained, linear_init_cfg, norm_init_cfg)
        if zero_init_output:
            last_linear = self.last_linear
            if last_linear is not None:
                nn.init.zeros_(last_linear.bias)
                last_linear.weight.data.copy_(0.01 * last_linear.weight.data)

    @property
    def last_linear(self):
        last_linear = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        return last_linear

    def forward(self, input, **kwargs):
        return self.mlp(input)

    def init_weights(self, pretrained=None, linear_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            linear_init = build_init(linear_init_cfg) if linear_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Linear) and linear_init:
                    linear_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError("pretrained must be a str or None")


@BACKBONES.register_module()
class ConvMLP(ExtendedModule):
    def __init__(
        self,
        mlp_spec,
        norm_cfg=dict(type="BN1d"),
        bias="auto",
        act_cfg=dict(type="ReLU"),
        inactivated_output=True,
        pretrained=None,
        conv_init_cfg=None,
        norm_init_cfg=None,
    ):
        super(ConvMLP, self).__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            if i == len(mlp_spec) - 2 and inactivated_output:
                act_cfg = None
                norm_cfg = None
            bias_i = need_bias(norm_cfg) if bias == "auto" else bias
            if norm_cfg is not None and norm_cfg.get("type") == "LN":
                self.mlp.add_module(f"conv{i}", nn.Conv1d(mlp_spec[i], mlp_spec[i + 1], 1, bias=bias_i))
                self.mlp.add_module(f"tranpose{i}-1", Rearrange("b c n -> b n c"))
                self.mlp.add_module(f"ln{i}", build_norm_layer(norm_cfg, num_features=mlp_spec[i + 1])[1])
                self.mlp.add_module(f"tranpose{i}-2", Rearrange("b n c -> b c n"))
                self.mlp.add_module(f"act{i}", build_activation_layer(act_cfg))
            else:
                self.mlp.add_module(
                    f"layer{i}",
                    ConvModule(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1,
                        bias=bias_i,
                        conv_cfg=dict(type="Conv1d"),
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=True,
                        with_spectral_norm=False,
                        padding_mode="zeros",
                        order=("dense", "norm", "act"),
                    ),
                )
        self.init_weights(pretrained, conv_init_cfg, norm_init_cfg)

    def forward(self, input, **kwargs):
        return self.mlp(input)

    def init_weights(self, pretrained=None, conv_init_cfg=None, norm_init_cfg=None):
        if isinstance(pretrained, str):
            logger = get_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            conv_init = build_init(conv_init_cfg) if conv_init_cfg else None
            norm_init = build_init(norm_init_cfg) if norm_init_cfg else None

            for m in self.modules():
                if isinstance(m, nn.Conv1d) and conv_init:
                    conv_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)) and norm_init:
                    norm_init(m)
        else:
            raise TypeError("pretrained must be a str or None")


# @BACKBONES.register_module()
# class ResidualConvMLP(nn.Module):
#     def __init__(self, mlp_spec, inactivated_output=True):
#         super(ResidualConvMLP, self).__init__()
#         self.mlp = nn.ModuleList()
#         self.ln = nn.ModuleList()
#         self.inactivated_output = inactivated_output
#         for i in range(len(mlp_spec) - 1):
#             self.mlp.append(nn.Conv1d(mlp_spec[i], mlp_spec[i + 1], 1, 1, 0, 1, 1, True))
#             self.mlp.append(nn.Conv1d(mlp_spec[i + 1], mlp_spec[i + 1], 1, 1, 0, 1, 1, True))
#             self.ln.append(nn.LayerNorm(mlp_spec[i + 1]))
#             self.ln.append(nn.LayerNorm(mlp_spec[i + 1]))

#     def forward(self, input):
#         ret = input
#         num_stages = len(self.mlp) // 2
#         for i in range(num_stages):
#             ret = self.ln[2 * i](self.mlp[2 * i](ret).transpose(1, 2)).transpose(1, 2)
#             ret = ret + self.mlp[i * 2 + 1](ret)
#             if not (i == num_stages - 1 and self.inactivated_output):
#                 ret = F.relu(self.ln[2 * i + 1](ret.transpose(1, 2)).transpose(1, 2))
#         return ret
