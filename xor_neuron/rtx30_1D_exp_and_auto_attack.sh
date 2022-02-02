#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/1D_xor_neuron/1D_xor_neuron_conv_cifar.yaml --exp_num 2 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/1D_xor_neuron/1D_xor_neuron_conv_cifar.yaml --exp_num 2 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/1D_xor_neuron/1D_xor_neuron_conv_cifar.yaml --exp_num 2 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done