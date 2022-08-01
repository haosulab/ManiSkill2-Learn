import math, time

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from torch import nn
import torch.nn.functional as F

from maniskill2_learn.utils.meta import Registry, build_from_cfg


LINEAR_LAYERS = Registry("linear layer")

for module in [nn.Identity, nn.Linear, nn.Bilinear, nn.LazyLinear]:
    LINEAR_LAYERS.register_module(module=module)


@LINEAR_LAYERS.register_module()
class EnsembledLinear(nn.Module):
    """EnsembledLinear
    Refer the linear module definition: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    """

    __constants__ = ["in_features", "out_features", "num_module"]
    in_features: int
    out_features: int
    num_modules: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, num_modules: int = 1, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(EnsembledLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules
        self.weight = Parameter(torch.zeros((num_modules, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(num_modules, out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight.data[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        ret = torch.einsum("bki,koi->bko", input, self.weight)
        if self.bias is not None:
            ret = ret + self.bias
        return ret

    def extra_repr(self) -> str:
        return "num_modules={}, in_features={}, out_features={}, bias={}".format(
            self.num_modules, self.in_features, self.out_features, self.bias is not None
        )


def build_linear_layer(cfg, default_args=None):
    return build_from_cfg(cfg, LINEAR_LAYERS, default_args)


'''
class LinearBlock(nn.Sequential):
    """Applies a 1D convolution over an input signal composed of several input planes,
    optionally followed by batch normalization and ReLU activation.
    """
    def __init__(self, linear_cfg=):
        pass


class LinearBNReLU(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes,
    optionally followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, relu=True, bn=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class LinearBlock(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes,
    optionally followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, relu=True, bn=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear = nn.Linear(in_channels, out_channels, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class Conv1dBNReLU(nn.Module):
    """Applies a 1D convolution over an input signal composed of several input planes,
    optionally followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2dBNReLU(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MLP(nn.ModuleList):
    def __init__(self, in_channels, mlp_channels, out_channels=None, bn=False, out_relu=True):
        """Multi-layer perception with relu activation
        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            bn (bool): whether to use batch normalization
        """
        super(MLP, self).__init__()
        if out_channels is not None:
            mlp_channels.append(out_channels)
        
        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        c_in = in_channels
        for ind, c_out in enumerate(mlp_channels):
            if ind==len(mlp_channels)-1:
                self.append(LinearBNReLU(c_in, c_out, relu=out_relu, bn=bn))
            else:
                self.append(LinearBNReLU(c_in, c_out, bn=bn))
            c_in = c_out

    def forward(self, x):
        for module in self:
            assert isinstance(module, LinearBNReLU)
            x = module(x)
        return x


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=True):
        """Multi-layer perception shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization

        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.ndim = ndim

        if ndim == 1:
            mlp_module = Conv1dBNReLU
        elif ndim == 2:
            mlp_module = Conv2dBNReLU
        else:
            raise ValueError('SharedMLP only supports ndim=(1, 2).')

        c_in = in_channels
        for ind, c_out in enumerate(mlp_channels):
            self.append(mlp_module(c_in, c_out, 1, relu=True, bn=bn))
            c_in = c_out

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1dBNReLU, Conv2dBNReLU))
            x = module(x)
        return x
'''
