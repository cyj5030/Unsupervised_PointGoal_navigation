#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 1 \
    rl_training.py \
    --exp-config configs/pointnav/pointnav.yaml \
    --run-type eval
