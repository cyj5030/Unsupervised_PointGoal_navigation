import torch
import torch.nn.functional as F
import numpy as np
import h5py
import pickle

import glob
import os
from tqdm import tqdm

class VODataset_small_chunk(torch.utils.data.Dataset):
    def __init__(self, path, run_type='train', num_frame=2, out_size=(256, 256)):
        self.out_size = out_size
        self.num_frame = num_frame
        self.run_type = run_type
        
        sequence_names = sorted(glob.glob(os.path.join(path, run_type, "small_chunk", '0*')))

        self.mapping_dict = {}
        map_id = 0
        for i, sn in tqdm(enumerate(sequence_names), desc="pre collect all data infomation."):
            filenames = sorted(glob.glob(os.path.join(sn, '*.pkl')))
            for j in range(num_frame-1, len(filenames)):
                self.mapping_dict[map_id] = (os.path.dirname(filenames[j]), j)
                map_id += 1

        self.length_ = len(self.mapping_dict)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        h5_data, iid = self.load_chunk(idx)
        outputs = self.pre_process(h5_data)

        outputs["iid"] = torch.tensor([iid])
        return outputs
    
    def load_chunk(self, idx):
        dirname, iid = self.mapping_dict[idx]

        data = {
            "rgb": [],
            "depth": [],
            "gps": [],
            "goal": [],
            "state": [],
            "action": [],
        }

        for i in range(-self.num_frame + 1, 1):
            # npz_data = np.load(os.path.join(dirname, str(iid+i).zfill(6) + ".npz"))
            with open(os.path.join(dirname, str(iid+i).zfill(6) + ".pkl"), 'rb') as f:
                npz_data = pickle.load(f)
            data["rgb"].append(npz_data["rgb"])
            data["depth"].append(npz_data["depth"])
            data["gps"].append(npz_data["gps"])
            data["goal"].append(npz_data["goal"])
            data["state"].append(npz_data["state"])
            data["action"].append(npz_data["action"])

        return data, iid
    
    def pre_process(self, data):
        outputs = {}
        outputs["rgb"] = torch.from_numpy(np.concatenate(data["rgb"], 2)).float().permute(2, 0, 1)  / 255.0
        outputs["depth"] = torch.from_numpy(np.concatenate(data["depth"], 2)).float().permute(2, 0, 1) 
        outputs["gps"] = torch.from_numpy(np.concatenate(data["gps"], 0)).float()
        outputs["goal"] = torch.from_numpy(np.concatenate(data["goal"], 0)).float()
        outputs["state"] = torch.from_numpy(np.concatenate(data["state"], 0)).float()
        outputs["action"] = torch.from_numpy(np.array(data["action"])).float()

        if outputs["rgb"].shape[1:3] != self.out_size:
            outputs["rgb"] = F.interpolate(outputs["rgb"][np.newaxis, ...], self.out_size, mode="bilinear")[0]
            outputs["depth"] = F.interpolate(outputs["depth"][np.newaxis, ...], self.out_size, mode="bilinear")[0]
        
        # outputs['rgbd'] = torch.cat([outputs["rgb"], outputs["depth"]], 0) # rgb rgb d d
        return outputs

class VODataset(torch.utils.data.Dataset):
    def __init__(self, path, run_type='train', num_frame=2, out_size=(256, 256)):
        self.out_size = out_size
        self.num_frame = num_frame
        self.run_type = run_type
        
        self.filenames = sorted(glob.glob(os.path.join(path, run_type, '*.hdf5')))

        mapping_dict_file = os.path.join(path, run_type, 'mapping_dict.pkl')
        if not os.path.exists(mapping_dict_file):
            self.mapping_dict = {}
            map_id = 0
            for i, fn in enumerate(self.filenames):
                with h5py.File(fn, "r") as f:
                    seq_range = os.path.basename(fn).split('.')[0].split('_')
                    # num_seq = int(seq_range[1]) - int(seq_range[0]) + 1
                    for j in range(int(seq_range[0]), int(seq_range[1]) + 1):
                        seq_name = "rgb_" + str(j).zfill(6)
                        seq_len = f[seq_name].shape[0]
                        for k in range(num_frame-1, seq_len):
                            self.mapping_dict[map_id] = (fn, seq_name, k)
                            map_id += 1
            with open(mapping_dict_file, 'wb') as pf:
                pickle.dump(self.mapping_dict, pf)
        else:
            with open(mapping_dict_file, 'rb') as pf:
                self.mapping_dict = pickle.load(pf)

        self.length_ = len(self.mapping_dict)

    def __len__(self):
        return self.length_

    def __getitem__(self, idx):
        h5_data, iid = self.load_h5(idx)
        outputs = self.pre_process(h5_data)

        outputs["iid"] = torch.tensor([iid])
        return outputs
    
    def load_h5(self, idx):
        filename, seq_name, iid = self.mapping_dict[idx]

        h5_data = {
            "rgb": [],
            "depth": [],
            "gps": [],
            "goal": [],
            "state": [],
            "action": [],
        }
        with h5py.File(filename, "r") as f:
            for i in range(-self.num_frame + 1, 1):
                h5_data["rgb"].append(f[seq_name][iid + i, :])
                h5_data["depth"].append(f[seq_name.replace("rgb", "depth")][iid + i])
                h5_data["gps"].append(f[seq_name.replace("rgb", "gps")][iid + i])
                h5_data["goal"].append(f[seq_name.replace("rgb", "goal")][iid + i])
                h5_data["state"].append(f[seq_name.replace("rgb", "state")][iid + i])
                h5_data["action"].append(f[seq_name.replace("rgb", "action")][iid + i])
        return h5_data, iid
    
    def pre_process(self, h5_data):
        outputs = {}
        outputs["rgb"] = torch.from_numpy(np.concatenate(h5_data["rgb"], 2)).float().permute(2, 0, 1)  / 255.0
        outputs["depth"] = torch.from_numpy(np.concatenate(h5_data["depth"], 2)).float().permute(2, 0, 1) 
        outputs["gps"] = torch.from_numpy(np.concatenate(h5_data["gps"], 0)).float()
        outputs["goal"] = torch.from_numpy(np.concatenate(h5_data["goal"], 0)).float()
        outputs["state"] = torch.from_numpy(np.concatenate(h5_data["state"], 0)).float()
        outputs["action"] = torch.from_numpy(np.array(h5_data["action"])).float()

        if outputs["rgb"].shape[1:3] != self.out_size:
            outputs["rgb"] = F.interpolate(outputs["rgb"][np.newaxis, ...], self.out_size, mode="bilinear")[0]
            outputs["depth"] = F.interpolate(outputs["depth"][np.newaxis, ...], self.out_size, mode="bilinear")[0]
        
        # outputs['rgbd'] = torch.cat([outputs["rgb"], outputs["depth"]], 0) # rgb rgb d d
        return outputs



if __name__ == "__main__":
    dataset = VODataset_small_chunk("./VO_data", "train_no_noise", num_frame=3)
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        if data["action"][0] == 0:
            print("i-1=0", dataset.mapping_dict[i][''])
        if data["action"][1] == 0:
            print("i=0", i)
        if data["action"][2] == 0:
            print("i+1=0", i)