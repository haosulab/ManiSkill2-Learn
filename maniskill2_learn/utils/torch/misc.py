from functools import wraps
import numpy as np
import torch
from maniskill2_learn.utils.math import split_num
from maniskill2_learn.utils.meta import get_logger
from maniskill2_learn.utils.data import DictArray, to_np, GDict, to_torch


def disable_gradients(network):
    for param in network.parameters():
        param.requires_grad = False


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed. Please try to be consistent.
    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed
    """
    base_seed = torch.IntTensor(1).random_().item()
    np.random.seed(base_seed + worker_id)


def no_grad(f):
    wraps(f)

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return wrapper


def get_seq_info(done_mask):
    # It will sort the length of the sequence to improve the performance

    # input: done_mask [L]
    # return: index [#seq, max_seq_len]; sorted_idx [#seq]; is_valid [#seq, max_seq_len]
    done_mask = to_np(done_mask)

    indices = np.where(done_mask)[0]
    one = np.ones(1, dtype=indices.dtype)
    indices = np.concatenate([one * -1, indices])
    len_seq = indices[1:] - indices[:-1]

    sorted_idx = np.argsort(-len_seq, kind="stable")  # From max to min
    max_len = len_seq[sorted_idx[0]]
    index = np.zeros([len(sorted_idx), max_len], dtype=np.int64)
    is_valid = np.zeros([len(sorted_idx), max_len], dtype=np.bool_)

    for i, idx in enumerate(sorted_idx):
        index[i, : len_seq[idx]] = np.arange(len_seq[idx]) + indices[idx] + 1
        is_valid[i, : len_seq[idx]] = True
    return index, sorted_idx, is_valid


def run_with_mini_batch(
    function,
    *args,
    batch_size=None,
    wrapper=True,
    device=None,
    ret_device=None,
    episode_dones=None,
    **kwargs,
):
    """
    Run a pytorch function with mini-batch when the batch size of dat is very large.
    :param function: the function
    :param data: the input data which should be in dict array structure
    :param batch_size: the num of samples in the whole batch
    :return: all the outputs.
    """
    capacity = None
    # print('In mini batch', batch_size, DictArray(kwargs).shape)

    def process_kwargs(x):
        if x is None or len(x) == 0:
            return None
        # print(type(x))
        nonlocal capacity, device, ret_device
        x = DictArray(x)
        # print(x.shape, x.type)
        # exit(0)
        capacity = x.capacity
        if device is None:
            device = x.one_device
        if ret_device is None:
            ret_device = x.one_device
        return x

    args, kwargs = list(args), dict(kwargs)
    # print(GDict(args).type)
    # print(GDict(kwargs).type)

    args = process_kwargs(args)
    kwargs = process_kwargs(kwargs)

    assert capacity is not None, "Input is None"
    if batch_size is None:
        batch_size = capacity

    ret = []
    # print(capacity, batch_size)
    for i in range(0, capacity, batch_size):
        num_i = min(capacity - i, batch_size)
        args_i = args.slice(slice(i, i + num_i)).to_torch(device=device, wrapper=False) if args is not None else []
        kwargs_i = kwargs.slice(slice(i, i + num_i)).to_torch(device=device, wrapper=False) if kwargs is not None else {}
        ret.append(GDict(function(*args_i, **kwargs_i)).to_torch(device=ret_device, wrapper=False))
    ret = DictArray.concat(ret, axis=0, wrapper=wrapper)

    return ret


def mini_batch(wrapper_=True):
    def actual_mini_batch(f):
        wraps(f)

        def wrapper(*args, batch_size=None, wrapper=None, device=None, ret_device=None, **kwargs):
            if wrapper is None:
                wrapper = wrapper_
            # print(batch_size, dict(kwargs))

            return run_with_mini_batch(f, *args, **kwargs, batch_size=batch_size, wrapper=wrapper, device=device, ret_device=ret_device)

        return wrapper

    return actual_mini_batch
