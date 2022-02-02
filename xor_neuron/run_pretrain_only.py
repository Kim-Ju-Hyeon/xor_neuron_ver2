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
from utils.arg_helper import get_config, edict2dict, mkdir

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
@click.option('--num', type=int, default=False)
def main(exp_path, num):
    if num == 1:
        seed_list = [1101, 1102, 1103, 1104, 1105, 1106]
    elif num == 2:
        seed_list = [1107,1108,1109,1110,1111,1112]
    elif num == 3:
        seed_list = [ 1113,1114,1115,1116,1117,1118]
    elif num == 4:
        seed_list = [1119,1120,1121,1122,35,65]
    else:
        seed_list = []

    for seed in seed_list:
        config = edict(yaml.load(open(exp_path, 'r'), Loader=yaml.FullLoader))

        config.exp_name = seed
        config.seed = seed

        config.exp_name = '_'.join([
            config.model.name, str(config.exp_name),
            time.strftime('%H%M%S')
        ])

        config.save_dir = os.path.join(config.exp_dir, config.exp_name)
        config.model_save = os.path.join(config.save_dir, "model_save")

        mkdir(config.exp_dir)
        mkdir(config.save_dir)
        mkdir(config.model_save)

        save_name = os.path.join(config.save_dir, 'config.yaml')
        yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

        config.seed = int(str(config.seed) + "001")

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        runner = eval(config.runner)(config)
        runner.pretrain(1)


    sys.exit(0)


if __name__ == "__main__":
    main()
