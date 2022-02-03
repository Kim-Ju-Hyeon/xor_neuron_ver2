import os
import sys
import time
import datetime
import torch
import logging
import traceback
import numpy as np
from pprint import pprint
import click

from easydict import EasyDict as edict
import yaml

from runner import *
from utils.slack import slack_message
from utils.logger import setup_logging
from utils.arg_helper import get_config

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
def main(exp_path):
    config = edict(yaml.load(open(exp_path, 'r'), Loader=yaml.FullLoader))

    config.save_dir = './time_check/model_save'
    config.model_save = './time_check/model_save'
    config.exp_dir = './time_check'
    config.train.max_epoch = 10

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(1))
    # logger = setup_logging('INFO', log_file)

    for layer_num in range(4):
        if 'Control' in config.model.name.split('_'):
            # if layer_num == 0:
            #     config.model.hidden_dim = [128]
            #
            # elif layer_num == 1:
            #     config.model.hidden_dim = [127,120]
            #
            # elif layer_num == 2:
            #     config.model.hidden_dim = [124,124,124]
            #
            # elif layer_num == 3:
            #     config.model.hidden_dim = [123, 119, 120, 120]

            if layer_num == 0:
                config.model.out_channel = [61, 30]
                config.model.kernel_size = [3, 1]
                config.model.zero_pad = [0, 0]
                config.model.stride = [1, 1]

            elif layer_num == 1:
                config.model.out_channel = [120, 120, 60]
                config.model.kernel_size = [3, 3, 1]
                config.model.zero_pad = [0, 0, 0]
                config.model.stride = [1, 1, 1]

            elif layer_num == 2:
                config.model.out_channel = [120, 150, 170, 170]
                config.model.kernel_size = [3, 3, 3, 1]
                config.model.zero_pad = [0, 0, 0, 0]
                config.model.stride = [1, 1, 1, 1]

            elif layer_num == 3:
                config.model.out_channel = [120, 180, 180, 320, 360]
                config.model.kernel_size = [3, 3, 2, 1, 1]
                config.model.zero_pad = [0, 0, 0, 0, 0]
                config.model.stride = [1, 1, 1, 1, 1]

            runner = eval(config.runner)(config)
            # print(f'{config.dataset.name} {config.model.name} {config.model.hidden_dim}')
            print(f'{config.dataset.name} {config.model.name} {config.model.out_channel}')
            runner.train_control()

        else:
            # if layer_num == 0:
            #     config.model.out_hidden_dim = [64]
            #
            # elif layer_num == 1:
            #     config.model.out_hidden_dim = [64,64]
            #
            # elif layer_num == 2:
            #     config.model.out_hidden_dim = [64,64,64]
            #
            # elif layer_num == 3:
            #     config.model.out_hidden_dim = [64,64,64,64]

            if layer_num == 0:
                config.model.out_channel = [30, 30]
                config.model.kernel_size = [3, 1]
                config.model.zero_pad = [0, 0]
                config.model.stride = [1, 1]

            elif layer_num == 1:
                config.model.out_channel = [120, 60, 60]
                config.model.kernel_size = [3, 3, 1]
                config.model.zero_pad = [0, 0, 0]
                config.model.stride = [1, 1, 1]

            elif layer_num == 2:
                config.model.out_channel = [60, 120, 120, 120]
                config.model.kernel_size = [3, 3, 3, 1]
                config.model.zero_pad = [0, 0, 0, 0]
                config.model.stride = [1, 1, 1, 1]

            elif layer_num == 3:
                config.model.out_channel = [60, 120, 180, 180, 360]
                config.model.kernel_size = [3, 3, 2, 1, 1]
                config.model.zero_pad = [0, 0, 0, 0, 0]
                config.model.stride = [1, 1, 1, 1, 1]

            runner = eval(config.runner)(config)
            # print(f'{config.dataset.name} {config.model.name} {config.model.out_hidden_dim}')
            print(f'{config.dataset.name} {config.model.name} {config.model.out_channel}')
            runner.train_phase1()


    sys.exit(0)


if __name__ == "__main__":
    main()
