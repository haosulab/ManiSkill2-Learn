import torch, torch.nn.functional as F
import torch.utils.cpp_extension
from torch import nn as nn
from maniskill2_learn.utils.torch import no_grad
import os.path as osp, time
from . import pcd_process_ext


__folder__ = osp.dirname(__file__)


# pcd_process_ext = torch.utils.cpp_extension.load(
#     name="pcd_process_ext",
#     sources=[osp.join(__folder__, "pcd_process.cpp"), osp.join(__folder__, "pcd_process.cu")],
#     extra_cflags=["-O3", "-std=c++17"],
#     verbose=True,
# )


def ravel_multi_index(multi_index: torch.Tensor):
    """
    Args:
        multi_index: [N, D] tensor.
    Returns:
        raveled_indices: [N] tensor.
    """
    min_index, max_index = multi_index.min(dim=0).values, multi_index.max(dim=0).values
    grid_size, multi_index = max_index - min_index + 1, multi_index - min_index

    grid_coef = torch.cumprod(grid_size, 0)
    grid_coef = torch.flip(grid_coef, [0])
    grid_coef = F.pad(grid_coef, [0, 1], value=1)[1:]
    print(grid_size, grid_coef)
    raveled_indices = (multi_index * grid_coef).sum(dim=-1)
    return raveled_indices


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


@no_grad
def downsample_pcd(xyz, mode="uniform", **kwargs):
    if mode == "maniskill":
        index, mask = pcd_process_ext.maniskill_downsample(xyz, **kwargs)
    elif mode == "voxel":
        index, mask = pcd_process_ext.voxel_downsample(xyz, **kwargs)
    elif mode == "uniform":
        index, mask = pcd_process_ext.uniform_downsample(xyz, **kwargs)
        """
        index = torch.randperm(xyz.shape[1], device=xyz.device)
        ret_index = torch.zeros([xyz.shape[0], num + 1], device=xyz.device, dtype=torch.long)
        ret_mask = torch.zeros([xyz.shape[0], num + 1], device=xyz.device, dtype=torch.bool)
        mask = xyz[:, index, -1] > min_z
        cum_idx = torch.cumsum(mask.long(), dim=-1)
        # mask = torch.logical_and(mask, cum_idx <= num)
        # final_idx = torch.where(mask, cum_idx - 1, num)
        # ret_index.scatter_(-1, final_idx, index.expand_as(xyz[..., 0]))
        # ret_mask.scatter_(-1, final_idx, mask)
        # ret_index = ret_index[:, :-1]
        # ret_mask = ret_mask[:, :-1]
        torch.cuda.synchronize()
        """
    return index, mask
