import numpy as np
from maniskill2_learn.utils.data import float_to_int, as_dtype, GDict, sample_and_pad, is_np
from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.version import __version__
import deprecation


def select_mask(obs, key, mask):
    if key in obs:
        obs[key] = obs[key][mask]


def pcd_filter_ground(pcd, eps=1e-3):
    return pcd["xyz"][..., -1] > eps


def pcd_filter_with_mask(obs, mask, env=None):
    assert isinstance(obs, dict), f"{type(obs)}"
    for key in ["xyz", "rgb", "seg"]:
        select_mask(obs, key, mask)


def pcd_uniform_downsample(obs, env=None, ground_eps=1e-3, num=1200):
    obs_mode = env.obs_mode
    assert obs_mode in ["pointcloud"]

    if ground_eps is not None:
        pcd_filter_with_mask(obs, pcd_filter_ground(obs, eps=ground_eps), env)
    pcd_filter_with_mask(obs, sample_and_pad(obs["xyz"].shape[0], num), env)
    return obs