import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *

import pytorch_lightning as pl
from torchmetrics import Accuracy

from utils.logger import get_logger


class PL_Runner(pl.LightningModule):

    def __init__(self, config, learning_rate):
        super().__init__()
        self.config = config
        self.model = eval(self.config.model.name)(self.config)
        self.acc_fnc = Accuracy()
        self.learning_rate = learning_rate
        self.my_logger = get_logger('exp_logger')

        self.train_conf = config.train
        self.model_conf = config.model
        self.dataset_conf = config.dataset

        self.best_val_loss = np.inf
        self.result = []

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)

        self.results['train_loss'] += [loss/x.shape[0]]

        metrics = {'train_loss': loss}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits, loss = self(x, y)
        acc = self.acc_fnc(logits, y)

        if loss < self.best_val_loss:
            self.best_val_loss = loss

        self.my_logger.info("Avg. Validation Loss = {:.6}".format(loss/x.shape[0], 0))
        self.my_logger.info("Current Best Validation Loss = {:.6}".format(self.best_val_loss))

        self.results['val_loss'] += [loss/x.shape[0]]
        self.results['val_acc'] += [acc]

        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits, loss = self(x)
        acc = self.acc_fnc(logits, y)

        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = optim.Adam(
            params,
            lr=self.learning_rate)

        self.my_logger.info(f"Learning Rate: {self.lr}")
        self.train_conf.lr = self.lr

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        return [optimizer], [scheduler]