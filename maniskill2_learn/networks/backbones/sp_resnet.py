"""Sparse Convolution.

References:
    https://github.com/mit-han-lab/torchsparse/blob/master/examples/example.py
    https://github.com/mit-han-lab/e3d/blob/master/spvnas/core/models/semantic_kitti/spvcnn.py
"""

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from pytorch3d.transforms import quaternion_to_matrix

from torchsparse import SparseTensor
import torchsparse.nn as spnn

from copy import copy


from maniskill2_learn.utils.torch import ExtendedModule
from .mlp import ConvMLP
from ..modules.weight_init import constant_init, kaiming_init
from ..modules.spconv_modules import initial_voxelize, voxel_to_point, point_to_voxel, ResidualBlock, Bottleneck, build_points, build_sparse_norm
from ..builder import BACKBONES


class SparseCNNBase(ExtendedModule):
    def preprocess(self, inputs, transpose=True, **kwargs):
        xyz = inputs["xyz"] if isinstance(inputs, dict) else inputs                   

        with torch.no_grad():
            if isinstance(inputs, dict):
                feature = [xyz]
                if "rgb" in inputs:
                    feature.append(inputs["rgb"])
                if "seg" in inputs:
                    feature.append(inputs["seg"])
                feature = torch.cat(feature, dim=-1)
            else:
                feature = xyz
            if transpose:
                feature = feature.transpose(1, 2).contiguous()
        return xyz, feature


@BACKBONES.register_module()
class SparseResNet(SparseCNNBase):
    BLOCK = None
    LAYERS = ()

    ARCH_SETTINGS = {
        6: (ResidualBlock, (1, 1, 0, 0)),
        10: (ResidualBlock, (1, 1, 1, 1)),  # (1 + 1 + 1 + 1) * 2 + 2
        18: (ResidualBlock, (2, 2, 2, 2)),
        34: (ResidualBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),  # 3 * (3 + 4 + 6 + 3) + 2
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        in_channel,
        voxel_size,
        out_channel=None,
        depth=10,
        cs=[64, 128, 256, 512],
        cr=1.0,
        dropout=0,
        zero_init_output=False,
        use_ln=True,
        **kwargs,
    ):
        super(SparseResNet, self).__init__()
        self.voxel_size = voxel_size
        self.out_channel = out_channel
        self.depth = int(eval(depth) if isinstance(depth, str) else depth)
        self.BLOCK, self.LAYERS = self.ARCH_SETTINGS[self.depth]

        assert len(cs) == 4
        cs = [int(cr * x) for x in cs]

        # In original resnet paper, stem network is single layer with kernel size 5. Here we use 2 layer with kernel size 3 instead!
        self.stem = nn.Sequential(
            spnn.Conv3d(in_channel, cs[0], kernel_size=3, stride=1),
            build_sparse_norm(cs[0], use_ln=use_ln),
            spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            build_sparse_norm(cs[0], use_ln=use_ln),
            spnn.ReLU(True),
        )
        in_channel = cs[0]
        self.net = nn.Sequential()
        for stage in range(len(cs)):
            if self.LAYERS[stage] > 0:
                self.net.add_module(f"block_{stage}_0", self.BLOCK(in_channel, cs[stage], kernel_size=3, stride=2, dilation=1, use_ln=use_ln))
                in_channel = cs[stage] * self.BLOCK.expansion
                for i in range(1, self.LAYERS[stage]):
                    self.net.add_module(f"block_{stage}_{i}", self.BLOCK(in_channel, cs[stage], kernel_size=3, stride=1, dilation=1, use_ln=use_ln))

        self.final_conv = nn.Sequential(
            spnn.Droupout(dropout, False),
            spnn.Conv3d(in_channel, in_channel, kernel_size=3, stride=3),
            build_sparse_norm(in_channel, use_ln=use_ln),
            spnn.ReLU(True),
        )

        self.pooling = spnn.GlobalMaxPool()
        self.feat_dim = in_channel
        self.regression = None
        if out_channel is not None:
            self.regression = nn.Sequential(nn.Linear(self.feat_dim, cs[-1] * 2), nn.ReLU(), nn.Linear(cs[-1] * 2, out_channel))
        self.init_weights()
        if zero_init_output:
            m = self.regression[-1]
            nn.init.zeros_(m.bias)
            m.weight.data.copy_(1e-3 * m.weight.data)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, spnn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m, 1)

    def forward(self, points, **kwargs):
        """
        xyz: [B, N, 3] or a batch of [N, 3]
        feature: [B, N, C] or a batch of [N, 3]
        """
        xyz, feature = self.preprocess(points, transpose=False, **kwargs)
        z = build_points(xyz, feature)

        x: SparseTensor = initial_voxelize(z, 1.0, self.voxel_size)
        x0 = self.stem(x)
        z0 = voxel_to_point(x0, z, nearest=False)
        x1 = point_to_voxel(x0, z0)
        x1 = self.net(x1)

        x2 = self.final_conv(x1)
        x2 = self.pooling(x2)
        if self.regression:
            x2 = self.regression(x2)
        return x2


@BACKBONES.register_module()
class SparseResNet6(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=6)


@BACKBONES.register_module()
class SparseResNet10(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=10)


@BACKBONES.register_module()
class SparseResNet18(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=18)


@BACKBONES.register_module()
class SparseResNet34(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=34)


@BACKBONES.register_module()
class SparseResNet50(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=50)


@BACKBONES.register_module()
class SparseResNet101(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=101)


@BACKBONES.register_module()
class SparseResNet152(SparseResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, depth=152)
