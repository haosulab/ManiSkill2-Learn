import math
import torch.nn as nn
import torch.nn.functional as F
from maniskill2_learn.utils.meta import Registry, build_from_cfg


CONV_LAYERS = Registry("conv layer")
for module in [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.LazyConv1d,
    nn.LazyConv2d,
    nn.LazyConv3d,
    nn.LazyConvTranspose1d,
    nn.LazyConvTranspose2d,
    nn.LazyConvTranspose3d,
    nn.Unfold,
    nn.Fold,
]:
    CONV_LAYERS.register_module(module=module)
CONV_LAYERS.register_module("Conv", module=nn.Conv2d)
CONV_LAYERS.register_module("Deconv", module=nn.ConvTranspose2d)

# SparseConv
SPARSE_CONV_LAYERS = Registry("sparse conv layer")
