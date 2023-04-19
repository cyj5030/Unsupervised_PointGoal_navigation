# Unsupervised Visual Odometry and Action Integration for PointGoal Navigation in Indoor Environment

Created by [Yijun Cao](https://github.com/cyj5030).

### [Paper](https://ieeexplore.ieee.org/abstract/document/10089465)
### [arxiv](https://arxiv.org/abs/2210.00413)

![Overall architecture](data/architecture.pdf)

## Citation
If you like our work and use the code or models for your research, please cite our work as follows.
```
@article{cao2023unsupervised,
  author={Cao, Yijun and Zhang, Xianshi and Luo, Fuya and Lin, Chuan and Li, Yongjie},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Unsupervised Visual Odometry and Action Integration for PointGoal Navigation in Indoor Environment}, 
  year={2023},
  doi={10.1109/TCSVT.2023.3263484}}
```

## Prerequisites
We only tested in environment: Ubuntu 18.04, CUDA 11.1, python 3.6.13, pytorch 1.8.0, habitat 0.2.1, habitat-sim 0.2.1.

The repo is tested under the following commits of [habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim).

```bash
habitat-lab == 0.2.1
habitat-sim == 0.2.1
```

## Data Download 
* [Gibson scene dataset](https://github.com/StanfordVL/GibsonEnv)

* PointGoal Navigation splits, we need [pointnav_gibson_v2.zip.](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets)

* Revise the item `SCENES_DIR` and `DATA_PATH` in configs/pointnav/pointnav_gibson.yaml to your dataset path.

## Data Preparation 
run collect_vo_data.py to generate VO data.
```bash
python collect_vo_data.py
```

## Inference Using Pretrained Model
We provide pretrained models [here](https://drive.google.com/drive/folders/1p-TfcibaFPPnuxf1YBX9XlOGXlIoFqyP?usp=share_link). The performance could be slightly different with the paper due to randomness.

To evaluate the prediction, download the pretrained models to train_log/{pointnav or vo}/checkpoints folder, and run

```bash
python rl_training.py --run-type eval
```

## Training
To reproduce the performance, we recommend that users try multiple training sessions.

1. train the VO using:
```bash
python vo_training.py 
```

2. train rl using:
```bash
python rl_training.py
```
Note that you should modify the `pretrain_path` in config file for better training, otherwise the training process will be very long.

## License

The codes and the pretrained model in this repository are under the BSD 2-Clause "Simplified" license as specified by the LICENSE file. 

## Related Projects
[The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation](https://github.com/Xiaoming-Zhao/PointNav-VO) 

