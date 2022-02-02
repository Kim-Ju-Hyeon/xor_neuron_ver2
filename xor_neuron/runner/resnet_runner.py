from __future__ import (division, print_function)
import os
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm
from collections import OrderedDict

import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard.writer import SummaryWriter

from scipy.signal import convolve2d
from scipy.stats import multivariate_normal

from model import *
from dataset.innernet_data import InnerNetData, InnerNetData_1D, InnerNetData_3D
from utils.slack import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, \
    EarlyStopper, save_outphase, make_mask
from utils.WarmupCosineLR import WarmupCosineLR
from utils.corpus import Corpus

from six.moves import urllib

logger = get_logger('exp_logger')
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# logger = get_logger('exp_logger')
EPS = float(np.finfo(np.float32).eps)
__all__ = ['ResnetRunner']


class ResnetRunner(object):

    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.cell_save = config.to_save_cell
        self.num_cell_types = config.model.num_cell_types
        self.pretrain_conf = config.pretrain
        self.logger = logger

    def pretrain(self, cell_type):
        print("Pretraining Start")
        print("-----------------------------------------------------------------")

        self.config.seed = int(str(self.seed) + str(cell_type))
        train_dataset = eval(self.dataset_conf.loader_name)(self.config, split='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.pretrain_conf.batch_size,
            shuffle=self.pretrain_conf.shuffle,
            num_workers=self.pretrain_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # Train innernet
        model = InnerNet_V2(self.config)
        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.pretrain_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.pretrain_conf.lr,
                momentum=self.pretrain_conf.momentum,
                weight_decay=self.pretrain_conf.wd)
        elif self.pretrain_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.pretrain_conf.lr,
                weight_decay=self.pretrain_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        # reset gradient
        optimizer.zero_grad()
        best_train_loss = np.inf
        for epoch in range(self.pretrain_conf.max_epoch):
            train_loss = []
            model.train()
            for xy, targets in train_loader:
                optimizer.zero_grad()

                if self.use_gpu:
                    xy, targets = xy.cuda(), targets.cuda()

                _, loss = model(xy, targets)
                loss.backward()
                optimizer.step()

                train_loss += [float(loss.data.cpu().numpy())]

            # display loss
            train_loss = np.stack(train_loss).mean()

            # save best model
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                print("Pretrain Loss @ epoch {:04d} = {:.6}".format(epoch + 1, np.mean(best_train_loss)))
                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_pretrained' + str(cell_type))

        return 1

    def train_phase1(self):
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])

        elif self.dataset_conf.name == 'cifar10':
            if self.dataset_conf.augmentation:
                transform = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])

            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)

            val_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                           train=False,
                                           transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]),
                                           download=True)

            # train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])

        elif self.dataset_conf.name == 'cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR100(root=self.dataset_conf.data_path,
                                              train=True,
                                              transform=transform,
                                              download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])

        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle,
                                  drop_last=True,
                                  )

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False,
                                drop_last=True,
                                )

        # create models
        model = ResNet20_Xor(self.config)
        load_model(model.inner_net, self.config.model_save + self.pretrain_conf.best_model[0])

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd,
                nesterov=True)

        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")


        total_steps = len(train_loader) * self.train_conf.max_epoch

        # lr_scheduler = WarmupCosineLR(
        #         optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
        #     )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        print("train_phase1 Start")
        print("-----------------------------------------------------------------")

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        best_val_acc = np.inf

        forward_time_list = []
        backward_time_list = []

        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            model.eval()
            val_loss = []
            correct = 0
            total = 0

            cnt = 0
            in2cells = []

            for imgs, labels in tqdm(val_loader):
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()
                with torch.no_grad():
                    out, loss, _ = model(imgs, labels)

                val_loss += [float(loss.data.cpu().numpy())]

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            val_loss = np.stack(val_loss).mean()
            val_acc = correct / total

            results['val_loss'] += [val_loss]
            results['val_acc'] += [correct / total]
            self.logger.info("Epoch {} Avg. Validation Loss = {:.6}".format(epoch, val_loss, 0))

            if self.config.to_save_cell:
                if (epoch < 20) or ((epoch + 1) % 25 == 0):
                    snapshot(
                        model.module.inner_net if self.use_gpu else model.inner_net,
                        optimizer,
                        self.config,
                        0,
                        tag=f'{epoch + 1}')

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = correct / total

                best_val_epoch = epoch

                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_phase1')

            if val_acc < best_val_acc:
                best_val_acc = val_acc

                snapshot(
                    model.module.inner_net if self.use_gpu else model.inner_net,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_acc_phase1')

            self.logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            # ====================== training ============================= #
            model.train()

            for imgs, labels in tqdm(train_loader):
                # 0. clears all gradients.
                optimizer.zero_grad()
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass

                # 2. compute loss
                forward_start = time.time()
                _, loss, _ = model(imgs, labels)
                forward_time = time.time() - forward_start

                # 3. backward pass (accumulates gradients).
                backward_start = time.time()
                loss.backward()
                backward_time = time.time() - backward_start
                # 4. performs a single update step.
                optimizer.step()

                train_loss = float(loss.data.cpu().numpy())
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    self.logger.info(
                        "Train Loss @ epoch {} iteration {} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1

            lr_scheduler.step()

        results['best_val_loss'] += [best_val_loss]
        results['best_val_acc'] += [best_val_acc]
        results['best_val_epoch'] = best_val_epoch

        forward_time_list = np.array(forward_time_list)
        backward_time_list = np.array(backward_time_list)

        time_dict = {'Forward': forward_time_list,
                     'Backward': backward_time_list}

        pickle.dump(time_dict, open(os.path.join(self.config.save_dir, 'time.p'), 'wb'))
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase1.p'), 'wb'))
        self.logger.info("Best Validation Loss = {:.6}".format(best_val_loss))

        return best_val_loss

    def train_phase2(self):
        print("train_phase2 Start")
        print("-----------------------------------------------------------------")
        self.train_conf.max_epoch = self.train_conf.max_epoch * 2

        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)
            train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])

        elif self.dataset_conf.name == 'cifar10':
            if self.dataset_conf.augmentation:
                transform = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
            train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])

        elif self.dataset_conf.name == 'cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR100(root=self.dataset_conf.data_path,
                                              train=True,
                                              transform=transform,
                                              download=True)
            train_dataset, val_dataset = random_split(train_dataset, [40000, 10000])

        else:
            raise ValueError("Non-supported dataset!")

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_conf.batch_size,
                                  shuffle=self.train_conf.shuffle)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_conf.batch_size,
                                shuffle=False)

        # create models
        model = ResNet20_Xor(self.config)

        # load inner-net trained on phase 1
        load_model(model.inner_net, self.config.model_save + self.train_conf.best_model)

        # ====== Freeze InnerNet  ====== #
        for child in model.children():
            if isinstance(child, nn.Sequential):
                for ch in child.children():
                    for chch in ch.children():
                        if isinstance(chch, nn.Sequential):
                            for param in chch.parameters():
                                param.requires_grad = False

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=self.train_conf.lr,
                momentum=self.train_conf.momentum,
                weight_decay=self.train_conf.wd,
                nesterov=True)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(
                params,
                lr=self.train_conf.lr,
                weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=10, is_decrease=False)

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.train_conf.lr_decay_steps,
            gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        if self.train_conf.is_resume:
            load_model(model, self.train_conf.resume_model, optimizer=optimizer)

        # ========================= Training Loop ============================= #
        iter_count = 0
        results = defaultdict(list)
        best_val_loss = np.inf
        best_val_acc = np.inf

        for epoch in range(self.train_conf.max_epoch):
            # ===================== validation ============================ #
            model.eval()
            val_loss = []
            correct = 0
            total = 0
            for imgs, labels in tqdm(val_loader):
                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                with torch.no_grad():
                    out, loss, _ = model(imgs, labels)
                val_loss += [float(loss.data.cpu().numpy())]

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            val_loss = np.stack(val_loss).mean()
            val_acc = correct / total
            results['val_loss'] += [val_loss]
            results['val_acc'] += [correct / total]
            self.logger.info("Avg. Validation Loss = {:.6} +- {}".format(val_loss, 0))

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = correct / total
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_phase2')

            if val_acc < best_val_acc:
                best_val_acc = val_acc
                snapshot(
                    model.module if self.use_gpu else model,
                    optimizer,
                    self.config,
                    epoch + 1,
                    tag='best_acc_phase2')

            self.logger.info("Current Best Validation Loss = {:.6}".format(best_val_loss))

            # ====================== training ============================= #
            model.train()
            for imgs, labels in tqdm(train_loader):
                # 0. clears all gradients.
                optimizer.zero_grad()

                if self.use_gpu:
                    imgs, labels = imgs.cuda(), labels.cuda()

                # 1. forward pass
                # 2. compute loss
                _, loss, _ = model(imgs, labels)
                # 3. backward pass (accumulates gradients).
                loss.backward()
                # 4. performs a single update step.
                optimizer.step()

                train_loss = float(loss.data.cpu().numpy())
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]

                # display loss
                if (iter_count + 1) % self.train_conf.display_iter == 0:
                    self.logger.info(
                        "Train Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count + 1, train_loss))

                iter_count += 1
            lr_scheduler.step()

        results['best_val_loss'] += [best_val_loss]
        results['best_val_acc'] += [best_val_acc]
        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats_phase2.p'), 'wb'))

        self.logger.info("Best Validation Loss = {:.6}".format(best_val_loss))

        return best_val_loss

    def test(self):
        print("Test Start")
        print("-----------------------------------------------------------------")
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=False)

        elif self.dataset_conf.name == 'cifar10':
            if self.dataset_conf.augmentation:
                transform = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
            test_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                            train=False,
                                            transform=transform,
                                            download=False)

        elif self.dataset_conf.name == 'cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            test_dataset = datasets.CIFAR100(root=self.dataset_conf.data_path,
                                             train=False,
                                             transform=transform,
                                             download=False)
        else:
            raise ValueError("Non-supported dataset!")

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.test_conf.batch_size,
                                 shuffle=False)

        # create models
        model = ResNet20_Xor(self.config)
        load_model(model, self.config.model_save + self.test_conf.test_model)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).cuda()

        model.eval()
        correct = 0
        total = 0
        in2cells = []
        cnt = 0

        for imgs, labels in tqdm(test_loader):
            if self.use_gpu:
                imgs, labels = imgs.cuda(), labels.cuda()

            with torch.no_grad():
                out, _, in2cells_per_layer_per_batch = model(imgs, labels)

                # # in2cells : num_layers x [data_size x num_cell_types x ... x arity]
                # for i, in2cells_per_layer in enumerate(in2cells_per_layer_per_batch):
                #     if cnt == 0:
                #         in2cells.append(in2cells_per_layer)
                #
                #     elif cnt < 100:
                #         in2cells[i] = np.concatenate((in2cells[i], in2cells_per_layer), 0)
                # cnt += 1

                _, pred = torch.max(out.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()

        test_accuracy = correct / total
        self.logger.info("Test Accuracy = {:.6} +- {}".format(test_accuracy, 0))

        pickle.dump(in2cells, open(os.path.join(self.config.model_save, 'in2cells.p'), 'wb'))

        return test_accuracy
