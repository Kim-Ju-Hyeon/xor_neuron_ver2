import os
import sys
import torch
import traceback
import numpy as np
import click
import pickle


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from runner import *
from dataset import *
from utils.logger import setup_logging
from utils.arg_helper import get_config

@click.command()
@click.option('--exp_path', type=str, default="./config/resnet/xor_resnet.yaml")
def main(exp_path):
    config = get_config(exp_path)
    config.seed = np.random.randint(10000)

    seed_everything(config.seed)
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    log_file = os.path.join(config.save_dir, "log_exp_{}.txt".format(config.seed))
    my_logger = setup_logging('INFO', log_file)
    my_logger.info("Writing log file to {}".format(log_file))
    my_logger.info("Exp instance id = {}".format(config.seed))

    # logger = WandbLogger(name=config.model.name, project=config.exp_name)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_last=False, dirpath=config.model_save,
                                 filename='model_snapshot_best_{epoch:d}')

    if config.use_gpu == 'True':
        gpu = 1
    else:
        gpu = 0

    try:
        trainer = Trainer(default_root_dir=config.save_dir,
                          max_epochs=config.train.max_epoch,
                          gpus=0,
                          auto_lr_find=True,
                          # logger=logger,
                          deterministic=True,
                          log_every_n_steps=1,
                          callbacks=[checkpoint])

        runner = PL_Runner(config)
        data = PL_DataModule(config)

        trainer.fit(runner, data)
        results = runner.results
        pickle.dump(results, open(os.path.join(config.save_dir, 'train_stats_phase1.p'), 'wb'))


    except:
        my_logger.error(traceback.format_exc())