"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    https://arxiv.org/abs/1612.00593
Reference Code:
    https://github.com/fxia22/pointnet.pytorch.git
"""

import numpy as np
from copy import deepcopy
import torch, torch.nn as nn, torch.nn.functional as F

from ..modules.attention import MultiHeadAttention
from .mlp import ConvMLP, LinearMLP
from ..builder import BACKBONES, build_backbone
from maniskill2_learn.utils.data import dict_to_seq, split_dim, GDict, repeat
from maniskill2_learn.utils.torch import masked_average, masked_max, ExtendedModule

from pytorch3d.transforms import quaternion_to_matrix


class STNkd(ExtendedModule):
    def __init__(self, k=3, mlp_spec=[64, 128, 1024], norm_cfg=dict(type="BN1d", eps=1e-6), act_cfg=dict(type="ReLU")):
        super(STNkd, self).__init__()
        self.conv = ConvMLP(
            [
                k,
            ]
            + mlp_spec,
            norm_cfg,
            act_cfg=act_cfg,
            inactivated_output=False,
        )  # k -> 64 -> 128 -> 1024
        pf_dim = mlp_spec[-1]
        mlp_spec = [pf_dim // 2**i for i in range(len(mlp_spec))]
        self.mlp = LinearMLP(mlp_spec + [k * k], norm_cfg, act_cfg=act_cfg, inactivated_output=True)  # 1024 -> 512 -> 256 -> k * k
        self.k = k

    def forward(self, feature):
        assert feature.ndim == 3, f"Feature shape {feature.shape}!"
        feature = self.mlp(self.conv(feature).max(-1)[0])
        feature = split_dim(feature, 1, [self.k, self.k])
        return torch.eye(self.k, device=feature.device) + feature


@BACKBONES.register_module()
class PointNet(ExtendedModule):
    def __init__(
        self,
        feat_dim,
        mlp_spec=[64, 128, 1024],
        global_feat=True,
        feature_transform=[
            1,
        ],
        norm_cfg=dict(type="LN1d", eps=1e-6),
        act_cfg=dict(type="ReLU"),
        num_patch=1,
    ):
        super(PointNet, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.num_patch = num_patch

        mlp_spec = deepcopy(mlp_spec)
        # Feature transformation in PointNet. For RL we usually do not use them.
        if 1 in feature_transform:
            self.stn = STNkd(3, mlp_spec, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if 2 in feature_transform:
            self.conv1 = ConvMLP([feat_dim, mlp_spec[0]], norm_cfg=norm_cfg, act_cfg=act_cfg, inactivated_output=False)
            self.fstn = STNkd(mlp_spec[0], mlp_spec, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.conv2 = ConvMLP(mlp_spec, norm_cfg=norm_cfg, act_cfg=act_cfg, inactivated_output=False)
        else:
            self.conv = ConvMLP(
                [
                    feat_dim,
                ]
                + mlp_spec,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inactivated_output=False,
            )

    def forward(self, inputs, object_feature=True, concat_state=None, **kwargs):
        xyz = inputs["xyz"] if isinstance(inputs, dict) else inputs

        if 1 in self.feature_transform:
            trans = self.stn(xyz.transpose(2, 1).contiguous())
            xyz = torch.bmm(xyz, trans)
        with torch.no_grad():
            if isinstance(inputs, dict):
                feature = [xyz]
                if "rgb" in inputs:
                    feature.append(inputs["rgb"])
                if "seg" in inputs:
                    feature.append(inputs["seg"])
                if concat_state is not None: # [B, C]
                    feature.append(concat_state[:, None, :].expand(-1, xyz.shape[1], -1))
                feature = torch.cat(feature, dim=-1)
            else:
                feature = xyz

            feature = feature.permute(0, 2, 1).contiguous()
        input_feature = feature
        if 2 in self.feature_transform:
            feature = self.conv1(feature)
            trans = self.fstn(feature)
            feature = torch.bmm(feature.transpose(1, 2).contiguous(), trans).transpose(1, 2).contiguous()
            feature = self.conv2(feature)
        else:
            feature = self.conv(feature)
        if self.global_feat:
            feature = feature.max(-1)[0]
        else:
            gl_feature = feature.max(-1, keepdims=True)[0].repeat(1, 1, feature.shape[-1])
            feature = torch.cat([feature, gl_feature], dim=1)

        return feature
