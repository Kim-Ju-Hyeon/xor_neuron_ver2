from runner import *
import pickle
import os
import yaml
from easydict import EasyDict as edict
from scipy.signal import convolve2d, fftconvolve, convolve
from scipy.stats import multivariate_normal
from model import *
from utils.train_helper import load_model

from utils.arg_helper import mkdir
import click
from glob import glob
import numpy as np
import torch
from utils.slack import slack_message, send_slack_message
import sys
import datetime
import traceback


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
@click.option('--attack_config', type=str, default="./config/adv_Attack/auto_attack_MNIST.yaml")
def auto_attack(exp_path, attack_config):
    attack_config = edict(yaml.load(open(attack_config, 'r'), Loader=yaml.FullLoader))
    config = edict(yaml.load(open(exp_path, 'r'), Loader=yaml.FullLoader))
    config.model.activation_fnc = 'ReLU'

    runner = eval(attack_config.runner)(config)
    if (attack_config.attack.name != "BoundaryAttack") and (attack_config.attack.name != "AutoAttack"):
        runner.gradient_base_attack(attack_config)
    elif attack_config.attack.name == "BoundaryAttack":
        runner.boundary_attack(attack_config)
    elif attack_config.attack.name == "AutoAttack":
        runner.auto_attack(attack_config)


if __name__ == '__main__':
    auto_attack()
