#!/bin/sh

#export CUDA_VISIBLE_DEVICES=0
#for i in 1
#do
#    python3 run_auto_attack.py --exp_path ../exp/CIFAR100/control_conv/Control_Conv_1_cifar100_0722/config.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
#    sleep 3
#done
#
#export CUDA_VISIBLE_DEVICES=1
#for i in 1
#do
#  python3 run_auto_attack.py --exp_path ../exp/CIFAR100/control_conv/Control_Conv_2_cifar100_0722/config.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
#  sleep 3
#done

export CUDA_VISIBLE_DEVICES=2
for i in 7
do
#    python3 run_auto_attack.py --exp_path ../exp/CIFAR100/xor_neuron_conv/ComplexNeuronConv_1_cifar100_0730/config.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
#    sleep 3
    python3 run_auto_attack.py --exp_path ../exp/CIFAR100/xor_neuron_conv/ComplexNeuronConv_2_cifar100_0730/config.yaml --attack_config ./config/adv_Attack/auto_attack_CIFAR10.yaml &
    sleep 3
done