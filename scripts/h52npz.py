import h5py
import numpy as np
import glob
import os
from tqdm import tqdm
import pickle

root = "./VO_data/train_no_noise"
out_root = os.path.join(root, "small_chunk")
if not os.path.exists(out_root):
    os.mkdir(out_root)

h5_filenames = sorted(glob.glob(os.path.join(root, "*.hdf5")))

for i, filename in tqdm(enumerate(h5_filenames)):
    out_data = {}
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            out_data[key] = f[key][:]

    seq_range = os.path.basename(filename).split('.')[0].split('_')
    for j in range(int(seq_range[0]), int(seq_range[1]) + 1):
        seq_len = out_data["rgb_" + str(j).zfill(6)].shape[0]
        for k in range(seq_len):
            data = {}
            data["rgb"] = out_data["rgb_" + str(j).zfill(6)][k]
            data["depth"] = out_data["depth_" + str(j).zfill(6)][k]
            data["gps"] = out_data["gps_" + str(j).zfill(6)][k]
            data["goal"] = out_data["goal_" + str(j).zfill(6)][k]
            data["state"] = out_data["state_" + str(j).zfill(6)][k]
            data["action"] = out_data["action_" + str(j).zfill(6)][k]

            out_foldername = os.path.join(out_root, str(j).zfill(6))
            if not os.path.exists(out_foldername):
                os.mkdir(out_foldername)
            out_filename = os.path.join(out_root, str(j).zfill(6), str(k).zfill(6))
            # np.savez(out_filename, **data)
            with open(out_filename + ".pkl", 'wb') as fw:
                pickle.dump(data, fw)
            
print()
