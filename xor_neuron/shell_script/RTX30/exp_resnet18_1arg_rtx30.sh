#!/bin/sh

#export CUDA_VISIBLE_DEVICES=0
#for i in 1
#do
#    python ../../run_resnet20.py --exp_path ../../config/resnet/0218_resnet50_xor_v3_rtx30/${i}.yaml &
#    sleep 3
#done

export CUDA_VISIBLE_DEVICES=1
for i in 1 2 3 4
do
    python ../../run_resnet20.py --exp_path ../../config/resnet/resnet18_1arg/${i}.yaml &&
    sleep 3
done
