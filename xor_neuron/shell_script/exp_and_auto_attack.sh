#!/bin/sh

#export CUDA_VISIBLE_DEVICES=2
#for i in 1
#do
#    python3 run_time_check.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_mlp_mnist.yaml &
##    python3 run_time_check.py --exp_path ./config/control_model/control_mlp_mnist.yaml &
#    sleep 3
#done


#export CUDA_VISIBLE_DEVICES=2
#for i in 1
#do
##    python3 run_time_check.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_mlp_cifar.yaml &
#    python3 run_time_check.py --exp_path ./config/control_model/control_mlp_cifar.yaml &
#    sleep 3
#done

#export CUDA_VISIBLE_DEVICES=2
#for i in 1
#do
##    python3 run_time_check.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_conv_mnist.yaml &
#    python3 run_time_check.py --exp_path ./config/control_model/control_conv_mnist.yaml &
#    sleep 3
#done


#export CUDA_VISIBLE_DEVICES=2
#for i in 1
#do
##    python3 run_time_check.py --exp_path ./config/2D_xor_neuron/2D_xor_neuron_conv_cifar.yaml &
#    python3 run_time_check.py --exp_path ./config/control_model/control_conv_cifar.yaml &
#    sleep 3
#done

#export CUDA_VISIBLE_DEVICES=0
#for i in 1
#do
#    python run_exp.py --exp_path ./config/resnet/xor_resnet.yaml &
#    sleep 3
#done

export CUDA_VISIBLE_DEVICES=3
for i in 1
do
    python run_exp.py --exp_path ./config/resnet/xor_resnet.yaml &
    sleep 3
done