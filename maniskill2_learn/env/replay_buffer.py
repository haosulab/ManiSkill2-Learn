import numpy as np
import os.path as osp
from typing import Union
from tqdm import tqdm
from itertools import count
from h5py import File

from maniskill2_learn.utils.meta import get_filename_suffix, get_total_memory, get_memory_list, get_logger, TqdmToLogger, parse_files
from maniskill2_learn.utils.data import is_seq_of, DictArray, GDict, is_h5, is_null, DataCoder, is_not_null
from maniskill2_learn.utils.file import load, load_items_from_record, get_index_filenames, get_total_size, FileCache, is_h5_traj, decode_items
from maniskill2_learn.utils.file.cache_utils import META_KEYS
from .builder import REPLAYS, build_sampling
from .sampling_strategy import TStepTransition


@REPLAYS.register_module()
class ReplayMemory:
    """
    This replay buffer is designed for RL, BRL.
    Replay buffer uses dict-array as basic data structure, which can be easily saved as hdf5 file.
    Also it utilize a asynchronized memory cache system to speed up the file loading process.
    See dict_array.py for more details.

    Two special keys for multiprocess and engineering usage
        is_truncated: to indicate if the trajectory is truncated here and the next sample from the same worker is from another trajectory.
        woker_index: to indicate which process generate this sample. Used in recurrent policy.
        is_valid: to indicate if this sample is useful. Used in recurrent policy.
    """

    def __init__(
        self,
        capacity,
        sampling_cfg=dict(type="OneStepTransition"),
        keys=None,
        keys_map=None,
        data_coder_cfg=None,
        buffer_filenames=None,
        cache_size=2048,
        num_samples=-1,
        num_procs=4,
        synchronized=True,  # For debug only which is slower than asynchronized file loading and data augmentation
        dynamic_loading=None,
        auto_buffer_resize=True,
        deterministic_loading=False,
    ):
        # capacity: the size of replay buffer, -1 means we will recompute the buffer size with files for initial replay buffer.
        assert capacity > 0 or buffer_filenames is not None
        # assert sampling_cfg is not None, "Please provide a valid sampling strategy over replay buffer!"

        if buffer_filenames is not None:
            logger = get_logger()
            buffer_filenames = parse_files(buffer_filenames)
            if deterministic_loading:
                logger.warning("Sort files and change sampling strategy!")
                sampling_cfg["no_random"] = True
                buffer_filenames = sorted(buffer_filenames)
        data_coder = None if is_null(data_coder_cfg) else DataCoder(**data_coder_cfg)
        if buffer_filenames is not None and len(buffer_filenames) > 0:
            logger.info(f"Load {len(buffer_filenames)} files!")
            data_size = get_total_size(buffer_filenames, num_samples=num_samples)
            self.data_size = data_size
            logger.info(f"Load {len(buffer_filenames)} files with {data_size} samples in total!")

            # For supervised learning with variable number of points.
            without_cache = data_coder is not None and data_coder.var_len_item
            # Cache utils does not support var input length recently
            if capacity < 0:
                capacity = data_size
                logger.info(f"Recomputed replay buffer size is {capacity}!")
            self.dynamic_loading = dynamic_loading if dynamic_loading is not None else (capacity < data_size)
            if self.dynamic_loading and cache_size != capacity:
                logger.warning("You should use same the cache_size as the capacity when dynamically loading files!")
                cache_size = capacity
            if not self.dynamic_loading and keys is not None:
                logger.warning("Some important keys may be dropped in buffer and the buffer cannot be extended!")
            if not without_cache:
                self.file_loader = FileCache(
                    buffer_filenames,
                    min(cache_size, capacity),
                    keys,
                    data_coder,
                    num_procs,
                    synchronized=synchronized,
                    num_samples=num_samples,
                    horizon=sampling_cfg.get("horizon", 1),
                    keys_map=keys_map,
                    deterministic_loading=deterministic_loading,
                )
                logger.info("Finish building file cache!")
            else:
                logger.info("Load without cache!")
        else:
            self.file_loader = None
            self.dynamic_loading = False
        if sampling_cfg is not None:
            sampling_cfg["capacity"] = capacity
            self.sampling = build_sampling(sampling_cfg)
        else:
            self.sampling = None

        if self.dynamic_loading:
            self.sampling.with_replacement = False

        self.capacity = capacity
        self.auto_buffer_resize = auto_buffer_resize
        self.memory = None
        self.position = 0
        self.running_count = 0
        self.reset()

        if buffer_filenames is not None and len(buffer_filenames) > 0:
            if self.dynamic_loading:
                self.file_loader.run(auto_restart=False)
                items = self.file_loader.get()
                self.push_batch(items)
                self.file_loader.run(auto_restart=False)
            else:
                logger.info("Load all the data at one time!")
                if not without_cache:
                    tqdm_obj = tqdm(file=TqdmToLogger(), mininterval=10, total=self.data_size)
                    while True:
                        self.file_loader.run(auto_restart=False)
                        items = self.file_loader.get()

                        if items is None:
                            break
                        self.push_batch(items)
                        tqdm_obj.update(len(items))
                else:
                    logger.info(f"Loading full dataset without cache system!")
                    for filename in tqdm(file=TqdmToLogger(), mininterval=60)(buffer_filenames):
                        file = File(filename, "r")
                        traj_keys = [key for key in list(file.keys()) if key not in META_KEYS]
                        if num_samples > 0:
                            traj_keys = traj_keys[:num_samples]
                        data = DictArray.from_hdf5(filename, traj_keys)
                        if keys is not None:
                            data = data.select_by_keys(keys)
                        if is_not_null(data_coder):
                            data = data_coder.decode(data)
                        data = data.to_two_dims()
                        self.push_batch(data)
                logger.info(f"Finish file loading! Buffer length: {len(self)}, buffer size {self.memory.nbytes_all / 1024 / 1024} MB!")
                logger.info(f"Len of sampling buffer: {len(self.sampling)}")

    def __getitem__(self, key):
        return self.memory[key]

    def __setitem__(self, key, value):
        self.memory[key] = value

    def __getattr__(self, key):
        return getattr(self.memory, key, None)

    def __len__(self):
        return min(self.running_count, self.capacity)

    def reset(self):
        self.position = 0
        self.running_count = 0
        # self.memory = None
        if self.sampling is not None:
            self.sampling.reset()

    def push(self, item):
        if not isinstance(item, DictArray):
            item = DictArray(item, capacity=1)
        self.push_batch(item)

    def push_batch(self, items: Union[DictArray, dict]):
        if not isinstance(items, DictArray):
            items = DictArray(items)
        if len(items) > self.capacity:
            items = items.slice(slice(0, self.capacity))

        if "worker_indices" not in items:
            items["worker_indices"] = np.zeros([len(items), 1], dtype=np.int32)
        if "is_truncated" not in items:
            items["is_truncated"] = np.zeros([len(items), 1], dtype=np.bool_)

        if self.memory is None:
            # Init the whole buffer
            self.memory = DictArray(items.slice(0), capacity=self.capacity)
        if self.position + len(items) > self.capacity:
            # Deal with buffer overflow
            final_size = self.capacity - self.position
            self.push_batch(items.slice(slice(0, final_size)))
            self.position = 0
            self.push_batch(items.slice(slice(final_size, len(items))))
        else:
            self.memory.assign(slice(self.position, self.position + len(items)), items)
            self.running_count += len(items)
            self.position = (self.position + len(items)) % self.capacity
            if self.sampling is not None:
                self.sampling.push_batch(items)

    def update_all_items(self, items):
        self.memory.assign(slice(0, len(items)), items)

    def tail_mean(self, num):
        return self.memory.slice(slice(len(self) - num, len(self))).to_gdict().mean()

    def get_all(self):
        # Return all elements in replay buffer
        return self.memory.slice(slice(0, len(self)))

    def to_hdf5(self, file, with_traj_index=False):
        data = self.get_all()
        if with_traj_index:
            # Save the whole replay buffer into one trajectory.
            # TODO: Parse the trajectories in replay buffer.
            data = GDict({"traj_0": data.memory})
        data.to_hdf5(file)

    def sample(self, batch_size, auto_restart=True, drop_last=True):
        if self.dynamic_loading and not drop_last:
            assert self.capacity % batch_size == 0

        batch_idx, is_valid = self.sampling.sample(batch_size, drop_last=drop_last, auto_restart=auto_restart and not self.dynamic_loading)
        if batch_idx is None:
            # without replacement only
            if auto_restart or self.dynamic_loading:
                items = self.file_loader.get()
                if items is None:
                    return None
                assert self.position == 0, "cache size should equals to buffer size"
                self.sampling.reset()
                self.push_batch(items)
                self.file_loader.run(auto_restart=auto_restart)
                batch_idx, is_valid = self.sampling.sample(batch_size, drop_last=drop_last, auto_restart=auto_restart and not self.dynamic_loading)
            else:
                return None
        ret = self.memory.take(batch_idx)
        ret["is_valid"] = is_valid
        return ret

    def mini_batch_sampler(self, batch_size, drop_last=False, auto_restart=False, max_num_batches=-1):
        if self.sampling is not None:
            old_replacement = self.sampling.with_replacement
            self.sampling.with_replacement = False
            self.sampling.restart()
        for i in count(1):
            if i > max_num_batches and max_num_batches != -1:
                break
            items = self.sample(batch_size, auto_restart, drop_last)
            if items is None:
                self.sampling.with_replacement = old_replacement
                break
            yield items

    def close(self):
        if self.file_loader is not None:
            self.file_loader.close()

    def __del__(self):
        self.close()
