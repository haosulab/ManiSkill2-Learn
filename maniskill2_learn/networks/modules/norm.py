import inspect, numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm

from maniskill2_learn.utils.data import is_tuple_of
from maniskill2_learn.utils.meta import Registry

NORM_LAYERS = Registry("norm layer")

NORM_LAYERS.register_module("BN", module=nn.BatchNorm2d)
NORM_LAYERS.register_module("SyncBN", module=nn.SyncBatchNorm)
NORM_LAYERS.register_module("BN1d", module=nn.BatchNorm1d)
NORM_LAYERS.register_module("BN2d", module=nn.BatchNorm2d)
NORM_LAYERS.register_module("BN3d", module=nn.BatchNorm3d)

NORM_LAYERS.register_module("GN", module=nn.GroupNorm)

NORM_LAYERS.register_module("IN", module=nn.InstanceNorm2d)
NORM_LAYERS.register_module("IN1d", module=nn.InstanceNorm1d)
NORM_LAYERS.register_module("IN2d", module=nn.InstanceNorm2d)
NORM_LAYERS.register_module("IN3d", module=nn.InstanceNorm3d)

NORM_LAYERS.register_module("LN", module=nn.LayerNorm)
NORM_LAYERS.register_module("LRN", module=nn.LocalResponseNorm)


NORM_LAYERS.register_module("LNkd")

"""
NOTE:
Different implementations of LayerNorm can have significant difference in speed:
https://github.com/pytorch/pytorch/issues/76012

nn.LayerNorm can be much slower than the custom LN version for the ConvNext model due to necessary 
permutation before / after the LN operation
"""

class LayerNormkD(LayerNorm):
    r"""Original implementation in PyTorch is not friendly for CNN which has channels_first manner.
    LayerNorm for CNN (1D, 2D, 3D)
    1D: [B, C, N]
    2D: [B, C, W, H]
    3D: [B, C, X, Y, Z]
    Modified from https://github.com/facebookresearch/ConvNeXt/blob/a9c608745273fa8419fe9e8ae3ce5bf8d32009dd/models/convnext.py
    """

    def __init__(self, *args, dim=1, data_format="channels_first", **kwargs):
        super(LayerNormkD, self).__init__(*args, **kwargs)
        assert data_format in ["channels_first", "channels_last"]
        self.dim = dim
        self.index_to_cl = (
            [
                0,
            ]
            + list(range(2, 2 + self.dim))
            + [
                1,
            ]
        )
        self.index_to_cf = [0, 1 + self.dim] + list(range(1, 1 + self.dim))
        self.data_format = data_format

    def forward(self, inputs):
        assert inputs.ndim == self.dim + 2 or (self.dim == 1 and inputs.ndim == 2)
        if self.data_format == "channels_last":
            return super(LayerNormkD, self).forward(inputs)
        else:
            if inputs.ndim > 2:
                inputs = inputs.permute(self.index_to_cl).contiguous()
            ret = super(LayerNormkD, self).forward(inputs)
            if inputs.ndim > 2:
                ret = ret.permute(self.index_to_cf).contiguous()
            return ret


@NORM_LAYERS.register_module("LN1d")
class LayerNorm1D(LayerNormkD):
    def __init__(self, *args, data_format="channels_first", **kwargs):
        super(LayerNorm1D, self).__init__(*args, **kwargs, data_format=data_format, dim=1)


@NORM_LAYERS.register_module("LN2d")
class LayerNorm2D(LayerNormkD):
    def __init__(self, *args, data_format="channels_first", **kwargs):
        super(LayerNorm2D, self).__init__(*args, **kwargs, data_format=data_format, dim=2)


@NORM_LAYERS.register_module("LN3d")
class LayerNorm3D(LayerNormkD):
    def __init__(self, *args, data_format="channels_first", **kwargs):
        super(LayerNorm3D, self).__init__(*args, **kwargs, data_format=data_format, dim=3)



class ConvNextLayerNorm(nn.Module):
    r""" 
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119

    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. 
    For 4D Tensor input, channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, *args, dim, eps=1e-6, data_format="channels_last", **kwargs):
        super().__init__()
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        if self.data_format == "channels_last":
            shape = args
        else:
            shape = list(args) + [1 for x in range(dim - len(args) + 1)]
        self.weight = nn.Parameter(torch.ones(*shape))
        self.bias = nn.Parameter(torch.zeros(*shape))
        self.eps = eps
        self.normalized_shape = shape
        self.dim = dim
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            if x.ndim == 2: # [B, C]
                assert self.dim == 1
                x = x[:, :, None]
            u = x.mean(1, keepdim=True)
            x = x - u
            s = x.pow(2).mean(1, keepdim=True)
            x = x / torch.sqrt(s + self.eps)
            #mean, var = torch.var_mean(x, dim=1, keepdims=True) # this consumes more mem than above
            #x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[None, ...] * x + self.bias[None, ...]
            x = x.squeeze(2)
            return x

    def __repr__(self):
        main_str = self._get_name() + '('
        main_str += f'Weight Shape: {self.normalized_shape}, dim: {self.dim}, data_format: {self.data_format}, eps: {self.eps}'
        main_str += ')'
        return main_str


@NORM_LAYERS.register_module("ConvNextLN1d")
class ConvNextLayerNorm1D(ConvNextLayerNorm):
    def __init__(self, *args, data_format="channels_first", **kwargs):
        super(ConvNextLayerNorm1D, self).__init__(*args, **kwargs, data_format=data_format, dim=1)


@NORM_LAYERS.register_module("ConvNextLN2d")
class ConvNextLayerNorm2D(ConvNextLayerNorm):
    def __init__(self, *args, data_format="channels_first", **kwargs):
        super(ConvNextLayerNorm2D, self).__init__(*args, **kwargs, data_format=data_format, dim=2)


@NORM_LAYERS.register_module("ConvNextLN3d")
class ConvNextLayerNorm3D(ConvNextLayerNorm):
    def __init__(self, *args, data_format="channels_first", **kwargs):
        super(ConvNextLayerNorm3D, self).__init__(*args, **kwargs, data_format=data_format, dim=3)




def need_bias(act_cfg):
    if act_cfg is None:
        return True
    if "BN" in act_cfg["type"] or "GN" in act_cfg["type"]:
        affine = act_cfg.get("affine", True)
    elif "LN" in act_cfg["type"]:
        affine = act_cfg.get("elementwise_affine", True)
    elif "IN" in act_cfg["type"]:
        affine = act_cfg.get("affine", False)
    elif "LRN" in act_cfg["type"]:
        affine = False
    else:
        raise TypeError(act_cfg["type"])
    return not affine


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(f"class_type must be a type, but got {type(class_type)}")
    if hasattr(class_type, "_abbr_"):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return "in"
    elif issubclass(class_type, _BatchNorm):
        return "bn"
    elif issubclass(class_type, nn.GroupNorm):
        return "gn"
    elif issubclass(class_type, nn.LayerNorm):
        return "ln"
    elif issubclass(class_type, nn.LocalResponseNorm):
        return "lrn"
    else:
        class_name = class_type.__name__.lower()
        if "batch" in class_name:
            return "bn"
        elif "group" in class_name:
            return "gn"
        elif "layer" in class_name:
            return "ln"
        elif "instance" in class_name:
            return "in"
        elif "local_response" in class_name:
            return "irn"
        else:
            return "norm_layer"


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        (str, nn.Module): The first element is the layer name consisting of abbreviation and postfix, e.g., bn1, gn.
            The second element is the created norm layer.
    """
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in NORM_LAYERS:
        raise KeyError(f"Unrecognized norm type {layer_type}")

    norm_layer = NORM_LAYERS.get(layer_type)
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
        if layer_type == "SyncBN":
            layer._specify_ddp_gpu_num(1)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def is_norm_layer(layer, exclude=None):
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude,)
        if not is_tuple_of(exclude, type):
            raise TypeError(f'"exclude" must be either None or type or a tuple of types, ' f"but got {type(exclude)}: {exclude}")

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm, nn.LocalResponseNorm)
    return isinstance(layer, all_norm_bases)
