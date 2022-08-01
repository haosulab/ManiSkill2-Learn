
import argparse
import os, numpy as np
import os.path as osp
import h5py
import glob

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from maniskill2_learn.utils.file import merge_h5_trajectory

"""
Example:
python tools/merge_h5.py --source-dir /TODO/demos/PickSingleYCB-v0/ --pattern "trajectory_pcd" \
--output-file /TODO/demos/PickSingleYCB-v0/trajectory_pcd_all.h5
"""

parser = argparse.ArgumentParser(description="Merge h5 files that match {pattern}.h5 under a directory into a single file")
parser.add_argument("--source-dir", type=str, default="")
parser.add_argument("--pattern", type=str, default="")
parser.add_argument("--output-file", type=str, default="")
args = parser.parse_args()
assert args.source_dir != "" and args.pattern != "" and args.output_file != ""

files = glob.glob(f'{args.source_dir}/**/{args.pattern}.h5', recursive=True)
print("Input files", files)
try:
    os.remove(args.output_file)
except:
    pass
merge_h5_trajectory(files, args.output_file)
