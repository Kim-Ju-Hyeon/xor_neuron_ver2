#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_resnet20.py --exp_path ./config/resnet/0126_resnet_xor_v2_DGX/${i}.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 2
do
    python run_resnet20.py --exp_path ./config/resnet/0126_resnet_xor_v2_DGX/${i}.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 3
do
    python run_resnet20.py --exp_path ./config/resnet/0126_resnet_xor_v2_DGX/${i}.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=0
for i in 4
do
    python run_resnet20.py --exp_path ./config/resnet/0126_resnet_xor_v2_DGX/${i}.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done