"""
IMPALA:
    Paper: Scalable Distributed Deep-RL with Importance Weighted Actor
    Reference code: https://github.com/facebookresearch/torchbeast

Nauture CNN:
    Code: https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/policies.py
"""


import numpy as np
import torch.nn as nn, torch, torch.nn.functional as F
from torch.nn import Conv2d
import math

from maniskill2_learn.networks.modules.weight_init import build_init
from maniskill2_learn.utils.data import GDict, get_dtype
from maniskill2_learn.utils.torch import ExtendedModule, no_grad, ExtendedSequential

from ..builder import BACKBONES
from ..modules import build_norm_layer, need_bias, build_activation_layer

class CNNBase(ExtendedModule):
    @no_grad
    def preprocess(self, inputs):
        # assert inputs are channel-first; output is channel-first
        if isinstance(inputs, dict):
            feature = []
            if "rgb" in inputs:
                # inputs images must not have been normalized before
                feature.append(inputs["rgb"] / 255.0)
            if "depth" in inputs:
                depth = inputs["depth"]
                if isinstance(depth, torch.Tensor):
                    feature.append(depth.float())
                elif isinstance(depth, np.ndarray):
                    feature.append(depth.astype(np.float32))
                else:
                    raise NotImplementedError()
            if "seg" in inputs:
                feature.append(inputs["seg"])
            feature = torch.cat(feature, dim=1)
        else:
            feature = inputs
        return feature


# @BACKBONES.register_module()
# class IMPALA(CNNBase):


@BACKBONES.register_module()
class IMPALA(CNNBase):
    def __init__(self, in_channel, image_size, out_feature_size=256, out_channel=None):
        super(IMPALA, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []
        in_channel = in_channel
        fcs = [64, 64, 64]

        self.stem = nn.Conv2d(in_channel, fcs[0], kernel_size=4, stride=4)
        in_channel = fcs[0]

        for num_ch in fcs:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            in_channel = num_ch
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.img_feat_size = math.ceil(image_size[0] / (2**len(fcs) * 4)) * math.ceil(image_size[1] / (2**len(fcs) * 4)) * fcs[-1]

        self.fc = nn.Linear(self.img_feat_size, out_feature_size)
        self.final = nn.Linear(out_feature_size, self.out_channel) if out_channel else None

    def forward(self, inputs, **kwargs):
        feature = self.preprocess(inputs)

        x = self.stem(feature)
        # x = feature
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.reshape(x.shape[0], self.img_feat_size)
        x = F.relu(self.fc(x))

        if self.final:
            x = self.final(x)

        return x


@BACKBONES.register_module()
class NatureCNN(CNNBase):
    # DQN
    def __init__(self, in_channel, image_size, mlp_spec=[32, 64, 64], out_channel=None, norm_cfg=dict(type="LN2d"), act_cfg=dict(type="ReLU"), **kwargs):
        super(NatureCNN, self).__init__()
        assert len(mlp_spec) == 3, "Nature Net only contain 3 layers"
        with_bias = need_bias(norm_cfg)
        self.net = ExtendedSequential(
            *[
                nn.Conv2d(in_channel, mlp_spec[0], 8, 4, bias=with_bias),
                build_norm_layer(norm_cfg, mlp_spec[0])[1],
                build_activation_layer(act_cfg),
                nn.Conv2d(mlp_spec[0], mlp_spec[1], 4, 2, bias=with_bias),
                build_norm_layer(norm_cfg, mlp_spec[1])[1],
                build_activation_layer(act_cfg),
                nn.Conv2d(mlp_spec[1], mlp_spec[2], 3, 1, bias=with_bias),
                build_norm_layer(norm_cfg, mlp_spec[2])[1],
                build_activation_layer(act_cfg),
                nn.Flatten(1),
            ]
        )
        with torch.no_grad():
            image = torch.zeros([1, in_channel] + list(image_size), device=self.device)
        feature_size = self.net(image).shape[-1]
        self.net.append_list(
            [
                nn.Linear(feature_size, 512),
                build_activation_layer(act_cfg),
            ]
        )
        if out_channel is not None:
            self.net.append_list([nn.Linear(512, 256), build_activation_layer(act_cfg), nn.Linear(256, out_channel)])

        if "conv_init_cfg" in kwargs:
            self.init_weights(self.convs, kwargs["conv_init_cfg"])
        

    def forward(self, inputs, **kwargs):
        feature = self.preprocess(inputs)
        return self.net(feature)

    def init_weights(self, conv_init_cfg=None):
        if conv_init_cfg is not None:
            init = conv_init_cfg(conv_init_cfg)
            init(self.convs)
