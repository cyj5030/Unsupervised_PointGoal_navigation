import numpy as np
import matplotlib.pylab as plt
import os
from scipy.optimize import curve_fit

colors = [
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.7098, 0.2000, 0.3608],
        [0.4902, 0.0706, 0.6863],
        [0.7059, 0.5333, 0.8824],
        [0.8000, 0.8000, 0.1000],
        [0.0588, 0.6471, 0.6471],
        [0.0392, 0.4275, 0.2667],
        [0.4157, 0.5373, 0.0824],
        [1.0000, 0.0000, 1.0000],
        [0.5490, 0.5490, 0.4549],
        [0.9373, 0.6863, 0.1255],
        [0.4471, 0.3333, 0.1725],
        [0.0000, 1.0000, 1.0000],
        [0.7176, 0.5137, 0.4392],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.7098, 0.2000, 0.3608],
    ]

def plot_save_lines(results, y_err, methods, save_filename, xylabel_title, 
                    x=None, xrange=None, yrange=None, linewidth=2, legend=False):
    linewidth = linewidth
    # xrange=(0.0,1.0)
    # yrange=(0.0,1.0)

    fig1 = plt.figure(1)
    if isinstance(results, list):
        length = len(results)
    else:
        length = results.shape[0]
    for i in range(length):
        if x is None:
            plt.plot(results[i], c=colors[i], linewidth=linewidth, label=methods[i])
        elif isinstance(x, tuple):
            x_r = np.linspace(x[0], x[1], results[i].shape[0])
            plt.plot(x_r, results[i], c=colors[i], linewidth=linewidth, label=methods[i])
        else:
            plt.plot(x, results[i], c=colors[i], linewidth=linewidth, label=methods[i])
            plt.fill_between(x, results[i] - y_err[i], results[i] + y_err[i], alpha=0.6)
    
    plt.tick_params(direction='in')
    if xrange is not None:
        plt.xlim(xrange[0],xrange[1])
        xyrange1 = np.arange(xrange[0], xrange[1]+0.01, 0.1)
        plt.xticks(xyrange1,fontsize=13,fontname='serif')

    if yrange is not None:
        plt.ylim(yrange[0],yrange[1])
        xyrange2 = np.arange(yrange[0], yrange[1]+0.01, 0.1)
        plt.yticks(xyrange2,fontsize=13,fontname='serif')
        

    plt.xlabel(xylabel_title[0],fontsize=16,fontname='serif')
    plt.ylabel(xylabel_title[1],fontsize=16,fontname='serif')
    plt.title(xylabel_title[2],fontsize=16,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 10,
    }
    if legend is not None:
        plt.legend(loc=legend, prop=font1)
    # plt.grid(linestyle='--')

    fig1.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

def load_log(filename):
    # ea = event_accumulator.EventAccumulator(filename)
    # ea.Reload()
    
    # required_data = []
    # for i, event in enumerate(ea.Scalars('reward')):
    #     # if event.step >= 2000:
    #     #     break
    #     required_data.append([event.step, event.value])

    with open(filename, "r") as f:
        data = f.readlines()

    required_data = []
    for line in data:
        if "update:" in line or "Average window size:" in line:
            required_data.append(line)
    
    logs = {
        "update": [],
        "frames": [],
        "distance_to_goal": [],
        "reward": [],
        "softspl": [],
        "spl": [],
        "success": [],
    }
    for iid in range(int(len(required_data) / 3)):
        temp0 = required_data[iid * 3 + 1][24:].strip().split("\t")
        temp1 = required_data[iid * 3 + 2][24:].strip()
        logs["update"].append(int(temp0[0].split(": ")[-1]))
        logs["frames"].append(int(temp0[-1].split(": ")[-1]))
        logs["distance_to_goal"].append(float(temp1.split("distance_to_goal: ")[-1][:5]))
        logs["reward"].append(float(temp1.split("reward: ")[-1][:5]))
        logs["softspl"].append(float(temp1.split("softspl: ")[-1][:5]))
        logs["spl"].append(float(temp1.split("spl: ")[-1][:5]))
        logs["success"].append(float(temp1.split("success: ")[-1][:5]))
        
    return logs

def fun2(x, a, b, c):
    return a*(x**2) + b*x + c

def fun_log(x, a, b, c):
    return a*np.log(b*x) + c

if __name__ == "__main__":
    methods = {
        1: ["pointnav_vo_irgbd_ttexture_3frame_cmodel_20epoch"],  # 1
        2: ["pointnav_gc_feature_wloss_detach_visual"], # 2
        3: ["pointnav_gc_feature_wloss_detach_visual_irgbtrgb"],
        4: ["pointnav_gc_feature_wloss_detach_visual_irgbdtrgbd"],
        5: ["pointnav_gc_feature_wloss_detach_visual_texture"],
        6: ["pointnav_gc_feature_woloss_detach_visual"],
        7: ["pointnav_gc_feature_wloss_detach_visual_detach_action"],
    }

    groups = {
        (1, 2, 6, 7): ["Embedding", "AIM", r"AIM only $\mathcal{L}_{vc}$", r"AIM w/o $\mathcal{L}_{vc}$"],
        (3, 4, 5): ["Visual (RGB), Rec. (RGB)", "Visual (RGBD), Rec. (RGBD)", "Visual (RGBD), Rec. (RGB-E)"],
        (5, 2): ["w/o CP-Net", "CP-Net"],
    }
    
    for key, value in methods.items():
        filename = os.path.join("./train_log", value[0], "train.log")
        logs = load_log(filename)
        methods[key].append(logs)

    plot_items = ["distance_to_goal", "reward", "softspl", "spl", "success"]
    total_steps = 2000
    for gid, (key, value) in enumerate(groups.items()):
        y = np.zeros((len(plot_items), len(key), total_steps))
        x = np.zeros((len(plot_items), len(key), total_steps))
        for i, mid in enumerate(plot_items):
            for j, pid in enumerate(key):
                y[i, j] = methods[pid][1][mid][:total_steps]
                x[i, j] = methods[pid][1]["update"][:total_steps]
        
        for i, mid in enumerate(plot_items):
            save_filename = os.path.join("./train_log/collection", f"nolegend_{gid:0>2d}_{plot_items[i]}.png")
            xylabel_title = ["update step", plot_items[i], ""]
            y_fit = np.zeros((len(key), total_steps))
            for j, pid in enumerate(key):
                popt, pcov = curve_fit(fun_log, x[i,0], y[i,j])
                y_fit[j] = fun_log(x[i,0], popt[0], popt[1], popt[2])
            y_err = np.abs(y_fit - y[i]) * 0.8

            plot_save_lines(y_fit, y_err, value, save_filename, xylabel_title, 
                            x=x[i, 0], xrange=None, yrange=None, linewidth=2, legend=None)
        print()
