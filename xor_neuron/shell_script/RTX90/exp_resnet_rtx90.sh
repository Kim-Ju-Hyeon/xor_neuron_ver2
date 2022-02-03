#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python run_resnet20.py --exp_path ./config/resnet/0126_resnet_xor_v2_rtx90/${i}.yaml &
    sleep 3
done