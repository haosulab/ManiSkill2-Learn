import torch.nn as nn
from maniskill2_learn.utils.meta import Registry, build_from_cfg

PADDING_LAYERS = Registry("padding layer")

for module in [
    nn.ReflectionPad1d,
    nn.ReflectionPad2d,
    nn.ReplicationPad1d,
    nn.ReplicationPad2d,
    nn.ReplicationPad3d,
    nn.ZeroPad2d,
    nn.ConstantPad1d,
    nn.ConstantPad2d,
    nn.ConstantPad3d,
]:
    PADDING_LAYERS.register_module(module=module)


def build_padding_layer(cfg, default_args=None):
    return build_from_cfg(cfg, PADDING_LAYERS, default_args)
