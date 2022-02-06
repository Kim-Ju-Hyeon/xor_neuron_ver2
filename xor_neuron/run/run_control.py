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
from utils.arg_helper import get_config, edict2dict

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
def main(exp_path, sample_id=1):
    config = get_config(exp_path, sample_id="{:03d}".format(sample_id))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging('INFO', log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))

    try:
        runner = eval(config.runner)(config)
        runner.train_control()

    except:
        logger.error(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
