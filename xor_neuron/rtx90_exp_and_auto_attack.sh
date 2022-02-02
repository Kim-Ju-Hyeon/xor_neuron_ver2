#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_conv_cifar100.yaml --exp_num 1 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_conv_cifar100.yaml --exp_num 1 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=0
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_conv_cifar100.yaml --exp_num 1 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 1
do
    python3 run_exp_and_auto_attack.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_conv_cifar100.yaml --exp_num 1 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done
