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

from runner import *
from utils.slack import slack_message
from utils.logger import setup_logging
from utils.arg_helper import get_config
from easydict import EasyDict as edict
import yaml


torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")

def main(exp_path, test=None, sample_id=1):
    config = edict(yaml.load(open(exp_path, 'r'), Loader=yaml.FullLoader))

    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(2))
    logger = setup_logging('INFO', log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(1))

    try:
        runner = ResnetRunner(config)
        runner.train_phase2()
        runner.test()

    except:
        logger.error(traceback.format_exc())


    sys.exit(0)


if __name__ == "__main__":
    main()
