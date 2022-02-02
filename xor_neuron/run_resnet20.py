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
@click.option('--attack_config', type=str, default="./config/adv_Attack/auto_attack_MNIST.yaml")
def main(exp_path, attack_config, test=None, sample_id=1):
    attack_config = edict(yaml.load(open(attack_config, 'r'), Loader=yaml.FullLoader))
    if test:
        config = edict(yaml.load(open(exp_path, 'r'), Loader=yaml.FullLoader))

    else:
        config = get_config(exp_path, sample_id="{:03d}".format(sample_id))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(1))
    logger = setup_logging('INFO', log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(1))

    try:
        runner = ResnetRunner(config)
        runner.pretrain(1)

        a = time.time()
        runner.train_phase1()
        print(f"It takes {time.time() - a}")
        runner.train_phase2()
        runner.test()

        attack_runner = eval(attack_config.runner)(config)
        attack_runner.auto_attack(attack_config)

    except:
        logger.error(traceback.format_exc())


    sys.exit(0)


if __name__ == "__main__":
    main()
