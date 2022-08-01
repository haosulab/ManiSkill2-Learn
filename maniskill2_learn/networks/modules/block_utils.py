import torch
from torch.functional import norm
from torch import nn

from .linear import LINEAR_LAYERS, build_linear_layer
from .conv import CONV_LAYERS, build_conv_layer
from .activation import build_activation_layer, INPLACE_ACTIVATIONS
from .padding import build_padding_layer
from .norm import build_norm_layer, need_bias
from .weight_init import kaiming_init, constant_init
from maniskill2_learn.utils.meta import Registry, build_from_cfg, ConfigDict


NN_BLOCKS = Registry("nn blocks")


@NN_BLOCKS.register_module()
class BasicBlock(nn.Sequential):
    def __init__(self, dense_cfg, norm_cfg=None, act_cfg=None, bias="auto", inplace=True, with_spectral_norm=False, order=("dense", "norm", "act")):
        super(BasicBlock, self).__init__()
        # dense here is conv or linear
        assert dense_cfg is None or isinstance(dense_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == set(["dense", "norm", "act"]), f"{order}"
        # assert dense_cfg.get(bias, True) or bias == 'auto', f"{dense_cfg.get(bias, None), bias}"

        self.dense_cfg = dense_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.with_spectral_norm = with_spectral_norm

        norm_follow_dense = order.index("norm") > order.index("dense")
        if dense_cfg.get("bias", bias) == "auto":
            dense_cfg["bias"] = need_bias(norm_cfg)
        dense_cfg.setdefault("bias", bias)

        if dense_cfg.get("type") in LINEAR_LAYERS:
            in_size = dense_cfg.get("in_features")
            out_size = dense_cfg.get("out_features")
        elif dense_cfg.get("type") in CONV_LAYERS:
            in_size = dense_cfg.get("in_channels")
            out_size = dense_cfg.get("out_channels")

        for name in order:
            if name == "dense":
                dense_type = dense_cfg.get("type", None)
                assert dense_type is not None and dense_type
                dense_cfg = dense_cfg.copy()

                if dense_type in CONV_LAYERS:
                    # Padding happen before convolution

                    official_conv_padding_mode = ["zeros", "reflect", "replicate", "circular"]  # Pytorch >= 1.7.1
                    padding_cfg = dense_cfg.pop("padding_cfg", None)
                    padding_mode = dense_cfg.get("padding_mode", None)
                    assert not (padding_cfg is None) or (padding_mode is None), "We only need one of padding_cfg and padding_mode"
                    if padding_cfg is not None:
                        padding_mode = padding_cfg.get("type", None)
                        if padding_mode is not None:
                            if padding_mode not in official_conv_padding_mode:
                                pad_cfg = dict(type=padding_mode)
                                self.add_module("padding", build_padding_layer(pad_cfg))
                            else:
                                dense_cfg["padding_mode"] = padding_mode
                    elif padding_mode is not None:
                        assert padding_mode in official_conv_padding_mode

                    layer = build_conv_layer(dense_cfg)
                    if self.with_spectral_norm:
                        layer = nn.utils.spectral_norm(layer)
                    self.add_module("conv", layer)
                elif dense_type in LINEAR_LAYERS:
                    self.add_module("linear", build_linear_layer(dense_cfg))
            elif name == "act" and act_cfg is not None:
                act_cfg = act_cfg.copy()
                # nn.Tanh has no 'inplace' argument
                if act_cfg["type"] in INPLACE_ACTIVATIONS:
                    act_cfg.setdefault("inplace", inplace)
                self.add_module("act", build_activation_layer(act_cfg))
            elif name == "norm" and norm_cfg is not None:
                if norm_follow_dense:
                    norm_channels = out_size
                else:
                    norm_channels = in_size
                norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
                self.add_module("norm", norm)
                self.norm_name = norm_name

        self.reset_parameters()

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def reset_parameters(self):
        # 1. It is mainly for customized conv layers with their own initialization manners by calling their own
        #    ``init_weights()``, and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization manners (that is, they don't have their own
        #   ``init_weights()``) and PyTorch's conv layers, they will be initialized by this method with default
        #   ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our initialization implementation using
        #     default ``kaiming_init``.
        for name, module in self.named_modules():
            if name in ["linear", "conv"]:
                if not hasattr(module, "reset_parameters"):
                    if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                        nonlinearity = "leaky_relu"
                        a = self.act_cfg.get("negative_slope", 0.01)
                    else:
                        nonlinearity = "relu"
                        a = 0
                    kaiming_init(module, a=a, nonlinearity=nonlinearity)
            elif name == "norm":
                constant_init(norm, 1, bias=0)


@NN_BLOCKS.register_module()
class FlexibleBasicBlock(BasicBlock):
    # The order of the operation is dynamic
    def forward(self, input, activate=True, norm=True):
        for name, module in self.named_modules():
            if (name == "act" and not activate) or (name == "norm" and not norm):
                input = module(input)
        return input


@NN_BLOCKS.register_module()
class ConvModule(BasicBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("dense", "norm", "act"),
    ):
        if conv_cfg is not None:
            conv_cfg = conv_cfg.copy()
        else:
            conv_cfg = ConfigDict(type="Conv2d", padding_mode=padding_mode)
        conv_cfg["in_channels"] = in_channels
        conv_cfg["out_channels"] = out_channels
        conv_cfg["kernel_size"] = kernel_size
        conv_cfg["stride"] = stride
        conv_cfg["padding"] = padding
        conv_cfg["dilation"] = dilation
        conv_cfg["groups"] = groups
        super(ConvModule, self).__init__(conv_cfg, norm_cfg, act_cfg, bias, inplace, with_spectral_norm, order)


@NN_BLOCKS.register_module()
class LinearModule(BasicBlock):
    def __init__(
        self,
        in_features,
        out_features,
        bias="auto",
        linear_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        order=("dense", "norm", "act"),
    ):
        if linear_cfg is not None:
            linear_cfg = linear_cfg.copy()
        else:
            linear_cfg = ConfigDict(type="Linear")
        linear_cfg["in_features"] = in_features
        linear_cfg["out_features"] = out_features
        super(LinearModule, self).__init__(linear_cfg, norm_cfg, act_cfg, bias, inplace, with_spectral_norm, order)


@NN_BLOCKS.register_module()
class MLP(nn.Sequential):
    def __init__(
        self,
        mlp_spec,
        in_features=None,
        out_features=None,
        block_cfg=dict(type="LinearModule", linear_cfg=dict(type="Linear"), norm_cfg=dict(type="BN1d"), act_cfg=dict(type="ReLU")),
        inactivated_output=True,
    ):
        super(MLP, self).__init__()
        block_cfg = block_cfg.copy()
        if in_features is not None:
            mlp_spec = [
                in_features,
            ] + mlp_spec
        if out_features is not None:
            mlp_spec = mlp_spec + [
                out_features,
            ]
        if block_cfg["linear_cfg"].get("type", "Linear") == "EnsembledLinear":
            self.ensenbled_model = True
            self.num_modules = block_cfg["linear_cfg"].get("num_modules", 1)
            if block_cfg.get("norm_cfg", None) is not None:
                print("Warning: if you want to use ensembled MLP with BN, " "please use multiple normal MLP instead!")
                block_cfg["norm_cfg"] = None
        else:
            self.ensenbled_model = False

        in_features = mlp_spec[0]
        for i in range(1, len(mlp_spec)):
            if i == len(mlp_spec) - 1:
                block_cfg["norm_cfg"] = None
                if inactivated_output:
                    block_cfg["act_cfg"] = None
            self.add_module(f"mlp_{i - 1}", build_nn_block(block_cfg, dict(in_features=in_features, out_features=mlp_spec[i])))
            in_features = mlp_spec[i]

    def forward(self, input):
        if self.ensenbled_model:
            # print(input.ndim, input.shape, self.num_modules)
            # print(input.ndim < 3, input.shape[1] != self.num_modules)
            # exit(0)
            assert input.ndim in [2, 3]
            if input.ndim == 2 or input.shape[1] != self.num_modules:
                input = torch.repeat_interleave(input[..., None, :], self.num_modules, dim=-2)
        return super(MLP, self).forward(input)


@NN_BLOCKS.register_module()
class SharedMLP(nn.Sequential):
    """
    Process data like PointCloud: [B, C, N]
    """

    def __init__(
        self,
        mlp_spec,
        in_features=None,
        out_features=None,
        block_cfg=dict(type="ConvModule", linear_cfg=dict(type="Conv1d"), norm_cfg=dict(type="BN1d"), act_cfg=dict(type="ReLU")),
        inactivated_output=True,
    ):
        super(SharedMLP, self).__init__()
        block_cfg = block_cfg.copy()
        if in_features is not None:
            mlp_spec = [
                in_features,
            ] + mlp_spec
        if out_features is not None:
            mlp_spec = mlp_spec + [
                out_features,
            ]
        in_features = mlp_spec[0]
        for i in range(1, len(mlp_spec)):
            if i == len(mlp_spec) - 1:
                block_cfg["norm_cfg"] = None
                if inactivated_output:
                    block_cfg["act_cfg"] = None
            self.add_module(f"mlp_{i - 1}", build_nn_block(block_cfg), dict(in_channels=in_features, kernel_size=1, out_channels=mlp_spec[i]))
            in_features = mlp_spec[i]


def build_nn_block(cfg, default_args=None):
    return build_from_cfg(cfg, NN_BLOCKS, default_args)
