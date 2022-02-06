import sys
import torch
import traceback
import numpy as np
import click

from easydict import EasyDict as edict
import yaml

from runner import *
from utils.logger import setup_logging
from utils.arg_helper import get_config

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="./config/resnet/xor_resnet.yaml")
def main(exp_path, sample_id=1):
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
        runner.train_phase1()
        runner.train_phase2()

    except:
        logger.error(traceback.format_exc())


    sys.exit(0)


if __name__ == "__main__":
    main()
