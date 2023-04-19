import os
import glob
import numpy as np
import shutil
import imageio.v3 as imageio
import tqdm

def video2png(video_path):
    # video_path = "/home/cyj/code/pointgoalnav_unsup_rgbd/train_log/pointnav_vo_irgbd_ttexture_3frame_cmodel_20epoch/videos"
    mp4s = sorted(glob.glob(os.path.join(video_path, "*.mp4")))
    # video_name = "scene=Cantwell.glb-episode=2-ckpt=45-distance_to_goal=0.33-success=1.00-spl=0.77-softspl=0.74-collisions.count=21.00.mp4"

    for video_name in tqdm.tqdm(mp4s):
        # vid = imageio.get_reader(os.path.join(video_name), "ffmepg")
        frames = imageio.imread(os.path.join(video_name), index=None)
        n_frames, h, w, _ = frames.shape

        # image_path = video_name.replace("videos", "images")[:-4]
        # if not os.path.exists(image_path):
        #     os.makedirs(image_path)

        frame_id = n_frames-1

        frame = frames[frame_id]
        
        try:
            idd = 500
            proj = (np.array(frame)/255.0).sum(0).sum(1)
            w_split = np.nonzero(proj[idd:] > 0.80*3*h)[0][0] + idd
            w_split=512
            td_map = frame[:, w_split:, :]
            rgbd = frame[:, :w_split, :]

        
            # imageio.imwrite(os.path.join(image_path, f"frame_id={frame_id}_rgbd.png"), rgbd)
            imageio.imwrite((video_name[:-4] + f"_{frame_id}_map.png"), td_map)
        except:
            print(f"faill in {video_name}")

proposed_path = "/home/cyj/code/pointgoalnav_unsup_rgbd_v2/train_log/pointnav_gc_feature_wloss_detach_visual/videos"
monodepth_path = "/home/cyj/code/pointgoalnav_unsup_rgbd_v2/train_log/pointnav_gc_feature_wloss_detach_visual_irgbtrgb/videos"
supVO_path = "/home/cyj/code/PointNav-VO/pretrained_ckpts/rl/tune_vo/seed_1-val-single_ckpt-train_noise_rgb_1_depth_1_act_1-eval_noise_rgb_1_depth_1_act_1-vo_rgb_d_dd_top_down_inv_joint-mode_det-rnd_n_10-20230210_185344779789/videos"
OA_path = "/home/cyj/code/OccupancyAnticipation/logs/videos"

infos = {
    "Ours": [proposed_path, {}],
    "Ours+monodepth": [monodepth_path, {}],
    "supVO": [supVO_path, {}],
    "OA": [OA_path, {}],
}

for key, Value in infos.items():
    filenames = sorted(glob.glob(os.path.join(Value[0], "*.mp4")))

    fn, skey, sr, spl, softspl = [], [], [], [], []
    for filename in filenames:
        items = os.path.basename(filename)[:-4].split('-')
        skey.append(items[0] + items[1])
        fn.append(filename.split('/')[-1])
        for item in items:
            key, value = item.split("=")
            if key == "success":
                sr.append(float(value))
            
            if key == "spl":
                spl.append(float(value))
            
            if key == "softspl":
                softspl.append(float(value))

    for i, j, m, k, f in zip(skey, sr, spl, softspl, fn):
        Value[1][i] = [j, m, k, f]

out_folder = "compare_sota_all_better_2"

keys = list(infos["Ours"][1])
for key in keys:
    if infos["Ours"][1][key][0] > infos["Ours+monodepth"][1][key][0] and \
       infos["Ours"][1][key][0] == infos["supVO"][1][key][0] and \
       infos["Ours"][1][key][0] > infos["OA"][1][key][0]:
    # if infos["Ours"][1][key][0] == infos["Ours+monodepth"][1][key][0] and \
    #    infos["Ours"][1][key][0] == infos["supVO"][1][key][0] and \
    #    infos["Ours"][1][key][0] == infos["OA"][1][key][0] and \
    #     infos["Ours"][1][key][0] == 1:
    # if infos["Ours"][1][key][0] > infos["Ours+monodepth"][1][key][0] and \
    #    infos["Ours"][1][key][0] > infos["supVO"][1][key][0] and \
    #    infos["Ours"][1][key][0] > infos["OA"][1][key][0]:
        print(key)
        shutil.copy(os.path.join(infos["Ours"][0], infos["Ours"][1][key][-1]), 
                    f"./train_log/pointnav_gc_feature_wloss_detach_visual/{out_folder}/" + infos["Ours"][1][key][-1].replace(".mp4", "-Ours.mp4"))
        shutil.copy(os.path.join(infos["Ours+monodepth"][0], infos["Ours+monodepth"][1][key][-1]), 
                    f"./train_log/pointnav_gc_feature_wloss_detach_visual/{out_folder}/" + infos["Ours+monodepth"][1][key][-1].replace(".mp4", "-Ours+monodepth.mp4"))
        shutil.copy(os.path.join(infos["supVO"][0], infos["supVO"][1][key][-1]), 
                    f"./train_log/pointnav_gc_feature_wloss_detach_visual/{out_folder}/" + infos["supVO"][1][key][-1].replace(".mp4", "-supVO.mp4"))
        shutil.copy(os.path.join(infos["OA"][0], infos["OA"][1][key][-1]), 
                    f"./train_log/pointnav_gc_feature_wloss_detach_visual/{out_folder}/" + infos["OA"][1][key][-1].replace(".mp4", "-OA.mp4"))

video2png(f"./train_log/pointnav_gc_feature_wloss_detach_visual/{out_folder}")