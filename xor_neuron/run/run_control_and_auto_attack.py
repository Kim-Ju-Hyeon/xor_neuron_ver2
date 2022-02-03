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
import numpy.random as npr
from easydict import EasyDict as edict
import yaml

from runner import *
from utils.slack import slack_message, send_slack_message
from utils.logger import setup_logging
from utils.arg_helper import get_config_for_multi_test

torch.set_printoptions(profile='full')


@click.command()
@click.option('--exp_path', type=str, default="./config/2D_xor_neuron/2D_xor_neuron_mlp_mnist.yaml")
@click.option('--exp_num', type=int, default=24)
@click.option('--attack_config', type=str, default="./config/adv_Attack/auto_attack_MNIST.yaml")
def main(exp_path, exp_num, attack_config):
    attack_config = edict(yaml.load(open(attack_config, 'r'), Loader=yaml.FullLoader))
    seed = npr.choice(exp_num * 5, size=exp_num, replace=False)

    for num in range(exp_num):
        start = datetime.datetime.now() + datetime.timedelta(hours=9)
        start_string = start.strftime('%Y-%m-%d %I:%M:%S %p')

        exp_name = num + 1
        config = get_config_for_multi_test(exp_path, seed[num], exp_name)

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.seed))
        logger = setup_logging('INFO', log_file, logger_name=f'exp_logger_{num}')
        logger.info("Writing log file to {}".format(log_file))
        logger.info("Exp instance id = {}".format(config.seed))

        try:
            runner = eval(config.runner)(config, logger)
            runner.train_control()

            end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')

            attack_runner = eval(attack_config.runner)(config)
            attack_runner.auto_attack(attack_config)

            end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')
            slack_message(start,
                          f"EXP Name: {config.exp_name} \n {attack_config.attack.name} Adversarial Attack Success \n "
                          f"Start at {start_string} \n End at {end_string}")

        except:
            logger.error(traceback.format_exc())
            end_string = (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y-%m-%d %I:%M:%S %p')
            slack_message(start,
                          f"EXP Name: {config.exp_name} \n Start at {start_string} \n End at {end_string} \n Error!!!!!")
            send_slack_message(traceback.format_exc())


if __name__ == "__main__":
    main()
