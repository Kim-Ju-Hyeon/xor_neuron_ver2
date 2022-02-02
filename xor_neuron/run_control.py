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
from utils.arg_helper import get_config, edict2dict

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="../exp/MLP_1D_arg/MNIST/")
@click.option('--test', type=bool, default=False)
def main(exp_path, test, sample_id=1):
    start = datetime.datetime.now() + datetime.timedelta(hours=9)
    start_string = start.strftime('%Y-%m-%d %I:%M:%S %p')

    if test:
        config = edict(yaml.load(open(exp_path, 'r'), Loader=yaml.FullLoader))

    else:
        config = get_config(exp_path, sample_id="{:03d}".format(sample_id))

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.run_id))
    logger = setup_logging(args.log_level, log_file)
    logger.info("Writing log file to {}".format(log_file))
    logger.info("Exp instance id = {}".format(config.run_id))

    try:
        runner = eval(config.runner)(config)

        if test:
            runner.test()

        else:
            runner.train_control()

        end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')
        slack_message(start,
                      f"EXP Name: {config.exp_name} \n Training Success \n Start at {start_string} \n End at {end_string}")

    except:
        logger.error(traceback.format_exc())
        end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')
        slack_message(start,
                      f"EXP Name: {config.exp_name} \n Start at {start_string} \n End at {end_string} \n Error!!!!!")
        send_slack_message(traceback.format_exc())

    sys.exit(0)


if __name__ == "__main__":
    main()
