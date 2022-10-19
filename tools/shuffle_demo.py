import h5py
import random
import argparse
import tqdm
import json
import copy
from multiprocessing import Pool


def copy_group(from_file, to_file, key_name, new_key_name):
    if new_key_name not in to_file.keys():
        to_file.require_group(new_key_name)
    for key in from_file[key_name].keys():
        new_data_key = key
        if "dist" in key or "str" in key:
            new_data_key = "_".join(key.split("_")[2:])
        if isinstance(from_file[key_name][key], h5py.Group):
            copy_group(from_file[key_name], to_file[new_key_name], key, new_data_key)
        else:
            to_file[new_key_name].create_dataset(
                new_data_key, data=from_file[key_name][key]
            )


parser = argparse.ArgumentParser()
parser.add_argument("--source-file", type=str)
parser.add_argument("--target-file", type=str)
args = parser.parse_args()

mapping = {}
source_file = h5py.File(args.source_file, "r")
target_file = h5py.File(args.target_file, "w")
order_list = list(range(len(list(source_file.keys()))))
random.shuffle(order_list)
for index, ori_index in tqdm.tqdm(enumerate(order_list), total=len(order_list)):
    old_key = "traj_" + str(ori_index)
    new_key = "traj_" + str(index)
    mapping[old_key] = new_key
    copy_group(source_file, target_file, old_key, new_key)
print(mapping)
source_file.close()
target_file.close()
