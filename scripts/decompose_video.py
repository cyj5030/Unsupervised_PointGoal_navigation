import imageio.v3 as imageio

import glob
import os
import numpy as np
import tqdm

if __name__ == "__main__":
    video_path = "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/pointnav_vo_irgbd_ttexture_3frame_cmodel_20epoch/videos"
    mp4s = glob.glob(os.path.join(video_path, "*.mp4"))
    # video_name = "scene=Cantwell.glb-episode=2-ckpt=45-distance_to_goal=0.33-success=1.00-spl=0.77-softspl=0.74-collisions.count=21.00.mp4"

    for video_name in tqdm.tqdm(mp4s):
        fps = imageio.immeta(os.path.join(video_name), exclude_applied=False)['fps']
        frames = imageio.imread(os.path.join(video_name), index=None)
        n_frames, h, w, _ = frames.shape

        image_path = video_name.replace("videos", "images")[:-4]
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        frame_ids = np.linspace(0, n_frames-1, 10).astype("int")

        for frame_id in frame_ids:
            frame = frames[frame_id]
            
            proj = (np.array(frame)/255.0).sum(0).sum(1)
            w_split = np.nonzero(proj[450:] > 0.95*3*h)[0][0] + 450

            td_map = frame[:, w_split:, :]
            rgbd = frame[:, :w_split, :]

            try:
                imageio.imwrite(os.path.join(image_path, f"frame_id={frame_id}_rgbd.png"), rgbd)
                imageio.imwrite(os.path.join(image_path, f"frame_id={frame_id}_map.png"), td_map)
            except:
                pass