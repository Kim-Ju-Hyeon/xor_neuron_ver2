import os
import glob
import torch
import pickle
import numpy as np
from collections import defaultdict
from scipy.signal import convolve2d, convolve
from scipy.stats import multivariate_normal

from torch.utils.data import Dataset
from utils.data_helper import *

__all__ = ['InnerNetData', 'InnerNetData_1D', 'InnerNetData_3D']

# For pretrain
class InnerNetData(Dataset):

  def __init__(self, config, split='train'):
    assert split in ['train', 'val', 'test'], "no such split"
    self.config = config
    self.split = split
    self.npr = np.random.RandomState(seed=config.seed)

    nb = 101
    x = np.linspace(-5, 5, nb)
    y = np.linspace(-5, 5, nb)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.reshape(-1), yv.reshape(-1)]).T
    mvn = multivariate_normal(mean=[0, 0], cov=[[1/9, 0], [0, 1/9]])
    gaussian_kernel = mvn.pdf(xy).reshape(nb, nb)
    gaussian_kernel /= gaussian_kernel.sum()
    init_unif = self.npr.uniform(-1, 1, size=(nb, nb))
    targets = convolve2d(init_unif, gaussian_kernel, mode='same').reshape(-1,1)

    self.xy = xy
    self.targets = targets

    pickle.dump(targets, open(os.path.join(self.config.save_dir, 'pretrain_target.p'), 'wb'))

  def __getitem__(self, index):
    return self.xy[index, None], self.targets[index, None]

  def __len__(self):
    return len(self.xy)

  def collate_fn(self, batch):
    assert isinstance(batch, list)

    xy_batch = torch.from_numpy(np.concatenate([bch for bch, _ in batch], axis=0)).float()
    targets_batch = torch.from_numpy(np.concatenate([bch for _, bch in batch], axis=0)).float()

    return xy_batch, targets_batch

class InnerNetData_1D(Dataset):

  def __init__(self, config, split='train'):
    assert split in ['train', 'val', 'test'], "no such split"
    self.config = config
    self.split = split
    self.npr = np.random.RandomState(seed=config.seed)

    nb = 101
    x = np.linspace(-5, 5, nb)
    mvn = multivariate_normal(mean=0, cov=1 / 9)
    gaussian_kernel = mvn.pdf(x)
    gaussian_kernel /= gaussian_kernel.sum()
    init_unif = self.npr.uniform(-1, 1, size=(nb))
    targets = convolve(init_unif, gaussian_kernel, mode='same').reshape(-1,1)

    self.x = x.reshape(-1,1)
    self.targets = targets

    pickle.dump(targets, open(os.path.join(self.config.save_dir, 'pretrain_target.p'), 'wb'))

  def __getitem__(self, index):
    return self.x[index, None], self.targets[index, None]

  def __len__(self):
    return len(self.x)

  def collate_fn(self, batch):
    assert isinstance(batch, list)

    xy_batch = torch.from_numpy(np.concatenate([bch for bch, _ in batch], axis=0)).float()
    targets_batch = torch.from_numpy(np.concatenate([bch for _, bch in batch], axis=0)).float()

    return xy_batch, targets_batch

class InnerNetData_3D(Dataset):
  def __init__(self, config, split='train'):
    assert split in ['train', 'val', 'test'], "no such split"
    self.config = config
    self.split = split
    self.npr = np.random.RandomState(seed=config.seed)

    nb = 101

    x = np.linspace(-5, 5, nb)
    y = np.linspace(-5, 5, nb)
    z = np.linspace(-5, 5, nb)
    xv, yv, zv = np.meshgrid(x, y, z)
    xyz = np.vstack([xv.reshape(-1), yv.reshape(-1), zv.reshape(-1)]).T
    mvn = multivariate_normal(mean=[0, 0, 0], cov=[[1 / 9, 0, 0], [0, 1 / 9, 0], [0, 0, 1 / 9]])
    gaussian_kernel = mvn.pdf(xyz).reshape(nb, nb, nb)
    gaussian_kernel /= gaussian_kernel.sum()
    init_unif = self.npr.uniform(-1, 1, size=(nb, nb, nb))
    targets = convolve(init_unif, gaussian_kernel, mode='same').reshape(-1, 1)

    self.xyz = xyz
    self.targets = targets

    pickle.dump(targets, open(os.path.join(self.config.save_dir, 'pretrain_target.p'), 'wb'))

  def __getitem__(self, index):
    return self.xyz[index, None], self.targets[index, None]

  def __len__(self):
    return len(self.xyz)

  def collate_fn(self, batch):
    assert isinstance(batch, list)

    xyz_batch = torch.from_numpy(np.concatenate([bch for bch, _ in batch], axis=0)).float()
    targets_batch = torch.from_numpy(np.concatenate([bch for _, bch in batch], axis=0)).float()

    return xyz_batch, targets_batch