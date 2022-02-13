#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
for i in 7 8
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/0213_resnet_xor_v2_DGX/${i}.yaml &&
    sleep 3
done
