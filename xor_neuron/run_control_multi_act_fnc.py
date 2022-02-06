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
from easydict import EasyDict as edict
import yaml
from runner import *
from utils.slack import slack_message
from utils.logger import setup_logging
from utils.arg_helper import get_config, get_config_for_multi_act_exp, edict2dict

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
@click.option('--act_fnc_list', type=click.STRING, default='ReLU, ELU')
@click.option('--exp_num', type=int, default=4)
def main(exp_path, act_fnc_list, exp_num):
    act_fnc_list = act_fnc_list.split(',')
    act_fnc_list = [act_fnc.strip() for act_fnc in act_fnc_list]
    activation_fnc_list = act_fnc_list

    try:
        for act_fnc in activation_fnc_list:
            seed = npr.choice(exp_num * 5, size=exp_num, replace=False)
            for num in range(exp_num):
                exp_name = num + 1

                config = get_config_for_multi_act_exp(exp_path, seed[num], exp_name, act_fnc)

                np.random.seed(config.seed)
                torch.manual_seed(config.seed)
                torch.cuda.manual_seed_all(config.seed)

                log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(1))
                logger = setup_logging('INFO', log_file)
                logger.info("Writing log file to {}".format(log_file))
                logger.info("Exp instance id = {}".format(config.seed))

                runner = eval(config.runner)(config)
                runner.train_control()

    except:
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
