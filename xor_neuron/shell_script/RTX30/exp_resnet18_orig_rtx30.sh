#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
for i in 1 2 3 4
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/resnet18_quad/${i}.yaml &&
    sleep 3
done
