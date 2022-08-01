import torch
import torch.nn as nn

from maniskill2_learn.utils.meta import build_from_cfg, Registry

ACTIVATION_LAYERS = Registry("activation layer")
for module in [
    nn.ELU,
    nn.Hardshrink,
    nn.Hardsigmoid,
    nn.Hardtanh,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.MultiheadAttention,
    nn.PReLU,
    nn.ReLU,
    nn.ReLU6,
    nn.RReLU,
    nn.SELU,
    nn.CELU,
    nn.GELU,
    nn.Sigmoid,
    nn.SiLU,
    #    nn.Mish,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.Tanh,
    nn.Tanhshrink,
    nn.Threshold,
    nn.Softmin,
    nn.Softmax,
    nn.Softmax2d,
    nn.LogSoftmax,
    nn.AdaptiveLogSoftmaxWithLoss,
]:
    ACTIVATION_LAYERS.register_module(module=module)


@ACTIVATION_LAYERS.register_module(name="Clip")
@ACTIVATION_LAYERS.register_module()
class Clamp(nn.Module):
    def __init__(self, min=-1.0, max=1.0):
        super(Clamp, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)


try:
    import torchsparse.nn as spnn

    ACTIVATION_LAYERS.register_module(name="SparseReLU", module=spnn.ReLU)
    ACTIVATION_LAYERS.register_module(name="SparseLeakyReLU", module=spnn.LeakyReLU)
except ImportError as e:
    print("Spconv is not installed!")
    print(e)


def build_activation_layer(cfg, default_args=None):
    return build_from_cfg(cfg, ACTIVATION_LAYERS, default_args)


INPLACE_ACTIVATIONS = ["ELU", "Hardsigmoid", "Hardtanh", "Hardswish", "ReLU", "LeakyReLU", "ReLU6", "RReLU", "SELU", "CELU", "SiLU", "Threshold"]
