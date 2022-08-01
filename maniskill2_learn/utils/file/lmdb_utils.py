import lmdb, os.path as osp, shutil, os, numpy as np
from .serialization import load, dump


class LMDBFile:
    def __init__(self, db_path, readonly=True, lock=True, all_async=False, readahead=False, replace=True, map_size=2 * (1024**4)):
        replace = replace and not readonly
        if replace:
            shutil.rmtree(db_path, ignore_errors=True)
            os.makedirs(db_path, exist_ok=True)
        self.env = lmdb.open(
            db_path,
            subdir=osp.isdir(db_path),
            map_size=map_size,
            readonly=readonly,
            metasync=all_async,
            sync=all_async,
            create=True,
            readahead=readahead,
            writemap=False,
            meminit=False,
            map_async=all_async,
            lock=lock,
        )

        self.readonly = readonly
        self.modified = False
        self.init_with_info = False
        self.writer = None
        self.reader = None
        self.length = len(self)

        if self.readonly:
            self.build_reader()
        else:
            self.build_writer()

    def __len__(self):
        return self.env.stat()["entries"]

    def close(self):
        if self.writer is not None:
            self.commit()
        self.env.sync()
        self.env.close()

    def build_writer(self):
        self.writer = self.env.begin(write=True)

    def build_reader(self):
        del self.reader
        self.reader = self.env.begin(write=False)

    def commit(self):
        self.writer.commit()
        self.writer = self.env.begin(write=True)

    def write_item(self, item):
        key = str(self.length)
        self.writer.put(key.encode(), dump(item, file_format="pkl"))
        self.length += 1
        if self.length % 100 == 0:
            self.commit()


#
#
# def max_abs_error(x, y):
#     return np.abs(x - y).max()
#
#
# def load_rgbd_trajectory(lmdb_folder='./robot_rgbd/'):
#     return LMDBDataset(lmdb_folder, process_function=process_sapien_rgbd_function)
#
#
# def process_h5_rgbd_trajectory(h5_files='/home/lz/data/Projects/PyTools/robot_rgbd.h5',
#                                lmdb_folder='./robot_rgbd/'):
#     # Use xuanlin's format
#     if isinstance(h5_files, str):
#         h5_files = [h5_files]
#
#     from maniskill2_learn.utils import load_trajectory
#     import pandas as pd
#
#     lmdb_file = LMDBFile(lmdb_folder, readonly=False, replace=True)
#
#     for h5_file in h5_files:
#         h5 = load_trajectory(h5_file)
#         keys = ['rewards', 'dones', 'episode_dones', 'state', 'next_state']
#         for h5_i in h5:
#             effective_len = pd.notnull(h5_i['dones']).sum()
#             for i in range(effective_len):
#                 ret = {}
#                 for key in keys:
#                     ret[key] = h5_i.loc[i, key]
#                 ret = compress_size(ret)
#                 for k1 in ['state', 'next_state']:
#                     for k2 in ['rgb', 'depth']:
#                         mode = list(ret[k1]['rgbd'].keys())[0]
#                         ret[k1]['rgbd'][mode][k2] = compress_image(ret[k1]['rgbd']['robot'][k2], depth=k2 == 'depth')
#                 lmdb_file.write_item(ret)
#         print(f'Finish {h5_file}, length of dataset {len(lmdb_file)}')
#     print('Total num of data', len(lmdb_file))
#     lmdb_file.close()
