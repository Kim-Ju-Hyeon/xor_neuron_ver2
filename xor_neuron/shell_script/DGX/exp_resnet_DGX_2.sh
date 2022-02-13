#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
for i in 3 4
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0213_resnet_xor_v2_DGX/${i}.yaml &&
    sleep 3
done
