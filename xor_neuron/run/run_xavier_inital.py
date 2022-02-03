from utils.arg_helper import get_config_for_multi_test
from runner import *
import pickle
import os
import yaml
from model import *
from utils.train_helper import load_model

from utils.arg_helper import mkdir
import click
from glob import glob
import numpy as np
import torch


@click.command()
@click.option('--exp_path', type=str, default="./config/Xavier_init/")
@click.option('--exp_num', type=int, default=24)
def xavier_inital(exp_path, exp_num):
    print(glob(exp_path + '*.yaml'))
    for file in sorted(glob(exp_path + '*.yaml')):
        print(file)
        for num in range(exp_num):
            seed = np.random.randint(1, 10000)
            seed2 = np.random.randint(10000, 20000)
            seed += seed2

            print(f'{num+1} exp')

            exp_name = num + 1

            config = get_config_for_multi_test(file, seed, exp_name)

            np.random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            print(f'seed {config.seed}')

            runner = eval(config.runner)(config)
            runner.train_phase1()
            # runner.test()


if __name__ == '__main__':
    xavier_inital()