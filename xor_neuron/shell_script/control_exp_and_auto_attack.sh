#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
for i in 1
do
    python3 run_control_and_auto_attack.py --exp_path ./config/control_model/control_mlp_mnist.yaml --exp_num 24 --attack_config ./config/adv_Attack/auto_attack_MNIST.yaml &
    python3 run_control_and_auto_attack.py --exp_path ./config/control_model/control_conv_cifar.yaml --exp_num 24 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &

    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python3 run_control_and_auto_attack.py --exp_path ./config/control_model/control_conv_mnist.yaml --exp_num 24 --attack_config ./config/adv_Attack/auto_attack_MNIST.yaml &
    python3 run_control_and_auto_attack.py --exp_path ./config/control_model/control_mlp_cifar.yaml --exp_num 24 --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &

    sleep 3
done