import torch
import torch.nn.functional as F
import numpy as np
import h5py

import glob
import os
from tqdm import tqdm

def h5_loader(filename, keys):
    outputs = {}
    with h5py.File(filename, "r") as f:
        for _key in keys:
            outputs[_key] = f[_key][:]
    return outputs

class GCDataset(torch.utils.data.Dataset):
    def __init__(self, path, run_type='train', preload=True):
        self.run_type = run_type
        self.preload = preload

        self.keys = ["pseudo_gps", "gps", "action", "goal", "state", "collision"]
        self.sequence_names = sorted(glob.glob(os.path.join(path, run_type, "*.hdf5")))

        if preload:
            self.data_info = []
            for i, _name in tqdm(enumerate(self.sequence_names), desc="pre load all data."):
                self.data_info.append( h5_loader(_name, self.keys) )
            self.length_ = len(self.data_info)
        else:
            self.length_ = len(self.sequence_names)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        if self.preload:
            data = self.data_info[idx]
        else:
            data = h5_loader(self.sequence_names[idx], self.keys)

        outputs = self.pre_process(data)

        return outputs
    
    def pre_process(self, data):
        outputs = {_key: torch.from_numpy(data[_key]) for _key in self.keys}
        return outputs


if __name__ == "__main__":
    dataset = GCDataset("./GC_data", "train", False)
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        print(data)