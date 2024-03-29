#!/bin/sh

# 'ReLU', 'ELU', 'LeakyReLU', 'PReLU', 'SELU', 'CELU', 'GELU', 'SiLU'
export CUDA_VISIBLE_DEVICES=0
for i in 2
do
   python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_mnist.yaml --act_fnc_list 'ReLU, ELU' --exp_num 4 &
   sleep 3
#   python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_cifar.yaml --act_fnc_list 'ReLU, ELU' --exp_num 4 &
#   sleep 3
done

export CUDA_VISIBLE_DEVICES=1
for i in 2
do
    python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_mnist.yaml --act_fnc_list 'LeakyReLU, PReLU' --exp_num 4 &
    sleep 3
#    python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_cifar.yaml --act_fnc_list 'LeakyReLU, PReLU' --exp_num 4 &
#    sleep 3
done


export CUDA_VISIBLE_DEVICES=2
for i in 2
do
    python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_mnist.yaml --act_fnc_list 'SELU, CELU' --exp_num 4 &
    sleep 3
#    python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_cifar.yaml --act_fnc_list 'SELU, CELU' --exp_num 4 &
#    sleep 3
done

export CUDA_VISIBLE_DEVICES=3
for i in 2
do
    python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_mnist.yaml --act_fnc_list 'GELU, SiLU' --exp_num 4 &
    sleep 3
#    python run_control_multi_act_fnc.py --exp_path ./config/control_model/control_mlp_cifar.yaml --act_fnc_list 'GELU, SiLU' --exp_num 4 &
#    sleep 3
done