#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0315_resnet_2arg_rtx30/${i}.yaml --exp_num 4 &
    sleep 3
done


export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0315_resnet_2arg_rtx30/${i}.yaml --exp_num 4 &
    sleep 3
done