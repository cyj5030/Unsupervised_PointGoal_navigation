import os
import numpy as np
import pickle
eval_methods = {
            "./train_log/pointnav_gc_feature_wloss_detach_visual_irgbtrgb/eval.log": [[349]],
            "./train_log/pointnav_gc_feature_wloss_detach_visual_irgbdtrgbd/eval.log": [[358]],
            "./train_log/pointnav_gc_feature_wloss_detach_visual_texture/eval.log": [[369]],
            "./train_log/pointnav_gc_feature_wloss_detach_visual/eval.log": [[339]],
            "./train_log/pointnav_gc_feature_woloss_detach_visual/eval.log": [[280]],
            "./train_log/pointnav_gc_feature_wloss_detach_visual_detach_action/eval.log": [[297]],
        }
# eval_methods = {
#         # "pointnav_gc_feature_wloss_detach_visual": ["vo_irgbd_ttexture_3frame_cmodel_20epoch", 280000, 339],
#         # "pointnav_gc_feature_wloss_detach_visual_irgbtrgb": ["vo_irgb_trgb_3frame", 175000, 349],
#         # "pointnav_gc_feature_wloss_detach_visual_irgbdtrgbd": ["vo_irgbd_trgbd_3frame", 170000, 358],
#         # "pointnav_gc_feature_wloss_detach_visual_texture": ["vo_irgbd_ttexture_3frame_20epoch", 280000, 369],
#         # "pointnav_gc_feature_woloss_detach_visual": ["vo_irgbd_ttexture_3frame_cmodel_20epoch", 280000, 280],
#         "pointnav_gc_feature_wloss_detach_visual_detach_action": ["vo_irgbd_ttexture_3frame_cmodel_20epoch", 280000, 297],
#     }
out_dir = "./train_log/collection/results.log"

def load_txt(path):
    with open(path, "r") as f:
        data = f.readlines()
    return data

def save_txt(path, data, mode):
    with open(path, mode) as f:
        f.writelines(data)

def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def get_match(data_str, match_str):
    reward, distance_to_goal, success, spl, softspl, length = [], [], [], [], [], 0.0
    for line_id, line in enumerate(data_str):
        if match_str in line:
            reward.append( float(data_str[line_id + 1].strip().split(":")[-1]) )
            distance_to_goal.append(float(data_str[line_id + 2].strip().split(":")[-1]) )
            success.append(float(data_str[line_id + 3].strip().split(":")[-1]) )
            spl.append(float(data_str[line_id + 4].strip().split(":")[-1]) )
            softspl.append(float(data_str[line_id + 5].strip().split(":")[-1]) )
            length += 1
    reward = np.array(reward)
    distance_to_goal = np.array(distance_to_goal)
    success = np.array(success)
    spl = np.array(spl)
    softspl = np.array(softspl)

    return reward, distance_to_goal, success, spl, softspl, length

def print_save(reward, distance_to_goal, success, spl, softspl, length, yaml_pth):
    mean_reward, std_reward = np.mean(reward), (np.max(reward) - np.min(reward))/2
    mean_distance_to_goal, std_distance_to_goal = np.mean(distance_to_goal), (np.max(distance_to_goal) - np.min(distance_to_goal))/2
    mean_success, std_success = np.mean(success), (np.max(success) - np.min(success))/2
    mean_spl, std_spl = np.mean(spl), (np.max(spl) - np.min(spl))/2
    mean_softspl, std_softspl = np.mean(softspl), (np.max(softspl) - np.min(softspl))/2

    lines = [
        f"{yaml_pth}, total length: {length}\n",
        f"Mean reward = {mean_reward:.3f}, std reward = {std_reward:.3f}\n",
        f"Mean distance_to_goal = {mean_distance_to_goal:.3f}, std distance_to_goal = {std_distance_to_goal:.3f}\n",
        f"Mean success = {mean_success:.3f}, std success = {std_success:.3f}\n",
        f"Mean spl = {mean_spl:.3f}, Mean spl = {std_spl:.3f}\n",
        f"Mean softspl = {mean_softspl:.3f}, Mean spl = {std_softspl:.3f}\n",
        "\n"
    ]
    # save_txt(out_dir, lines)
    print(f"{yaml_pth}, total length: {length}")
    print(f"Mean reward = {mean_reward:.3f}, std reward = {std_reward:.3f}")
    print(f"Mean distance_to_goal = {mean_distance_to_goal:.3f}, std distance_to_goal = {std_distance_to_goal:.3f}")
    print(f"Mean success = {mean_success:.3f}, std success = {std_success:.3f}")
    print(f"Mean spl = {mean_spl:.3f}, Mean spl = {std_spl:.3f}")
    print(f"Mean softspl = {mean_softspl:.3f}, Mean spl = {std_softspl:.3f}")
    return lines

if __name__ == "__main__":
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))

    id = 0
    for yaml_pth, _ckpts in eval_methods.items():
        for ckpts in _ckpts:
            if len(ckpts) == 1:
                rl_ckpt = ckpts[0]
                match_str = f"checkpoints {rl_ckpt}"
            else:
                vo_ckpt, rl_ckpt = ckpts
                match_str = f"rl_checkpoints {rl_ckpt}, vo_checcckpoints {vo_ckpt}"
            # vo info
            # vo_data = load_pickle(os.path.join(os.path.dirname(yaml_pth), "rl_vo_infos.pkl"))

            # rl info
            
            data_str = load_txt(yaml_pth)
            reward, distance_to_goal, success, spl, softspl, length = get_match(data_str, match_str)

            lines = print_save(reward, distance_to_goal, success, spl, softspl, length, yaml_pth)

            if id == 0:
                save_txt(out_dir, lines, "w")
            else:
                save_txt(out_dir, lines, "a")
            id += 1
