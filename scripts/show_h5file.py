from genericpath import exists
import h5py
import cv2
import os

h5filename = './VO_data/train/000037_000050.hdf5'
out_path = "./VO_data/shows"

def save_png(data, prefix, scale=1.0):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for i in range(data.shape[0]):
        cv2.imwrite(os.path.join(out_path, '%s_%d.png' % (prefix, i)), data[i]*scale)

if __name__ == "__main__":
    with h5py.File(h5filename, "r") as f:
        keys = [k for k in f.keys() if "rgb" in k]
        for iid, k in enumerate(keys):
            save_png(f[k][:], k)
            save_png(f[k.replace("rgb", "depth")][:], k.replace("rgb", "depth"), scale=255)