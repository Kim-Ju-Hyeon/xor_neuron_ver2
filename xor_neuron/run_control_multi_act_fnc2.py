import os
import sys
import time
import datetime
import torch
import logging
import traceback
import numpy as np
import numpy.random as npr
from pprint import pprint
import click

from runner import *
from utils.slack import slack_message
from utils.logger import setup_logging
from utils.arg_helper import get_config, get_config_for_multi_act_exp, edict2dict

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
@click.option('--exp_num', type=int, default=4)
def main(exp_path, exp_num):
    activation_fnc_list = ['SELU', 'CELU', 'GELU', 'SiLU']

    for act_fnc in activation_fnc_list:
        seed = npr.choice(exp_num * 5, size=exp_num, replace=False)
        for num in range(exp_num):
            exp_name = num + 1

            config = get_config_for_multi_act_exp(exp_path, seed[num], exp_name, act_fnc)

            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

            runner = eval(config.runner)(config)
            runner.train_control()

if __name__ == "__main__":
    main()
