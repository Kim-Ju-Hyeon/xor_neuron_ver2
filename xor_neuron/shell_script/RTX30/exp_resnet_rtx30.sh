#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0207_resnet_quad_v2_rtx30/${i}.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 2
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0207_resnet_quad_v2_rtx30/${i}.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 3
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0207_resnet_quad_v2_rtx30/${i}.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=0
for i in 4
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0207_resnet_quad_v2_rtx30/${i}.yaml &
    sleep 3
done