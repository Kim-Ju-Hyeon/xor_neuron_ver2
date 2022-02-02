import os
import torch
from model import *

from scipy.signal import convolve2d, convolve
from scipy.stats import multivariate_normal


def data_to_gpu(*input_data):
  return_data = []
  for dd in input_data:
    if type(dd).__name__ == 'Tensor':
      return_data += [dd.cuda()]
  
  return tuple(return_data)

def make_mask(in2cells, gaussian_kernel, arg_in_dim):
    if arg_in_dim == 2:
        in2cells = in2cells[0]
        in2cells = np.array(in2cells)
        in2cells = np.moveaxis(in2cells, -1, 0)
        in2cells = in2cells.reshape((2, -1))
        xedges = yedges = np.arange(-5.05, 5.1, 0.1)
        pdf, _, _ = np.histogram2d(in2cells[0], in2cells[1], bins=(xedges, yedges))
        pdf = convolve2d(pdf, gaussian_kernel, mode='same')
        pdf /= sum(pdf.flatten())

        threshold = 0.0005
        while True:
            row, col = np.where(pdf > threshold)
            if sum(pdf[row, col]) > 0.9:
                break
            else:
                threshold -= 0.00001
        mask = np.empty((101, 101))
        mask[:] = np.nan
        mask[row, col] = 1

    elif arg_in_dim == 1:
        in2cells = in2cells[0]
        in2cells = np.array(in2cells)
        in2cells = np.moveaxis(in2cells, -1, 0)

        xedges = np.arange(-5.05, 5.1, 0.1)
        pdf, _ = np.histogram(in2cells[0], bins=xedges)
        pdf = convolve(pdf, gaussian_kernel, mode='same')
        pdf /= sum(pdf.flatten())

        threshold = 0.0005
        while True:
            idx = np.where(pdf > threshold)
            if sum(pdf[idx]) > 0.9:
                break
            else:
                threshold -= 0.00001
        mask = np.empty(101)
        mask[:] = np.nan
        mask[idx] = 1

    return mask, pdf

def snapshot(model, optimizer, config, step, gpus=[0], tag=None):
  model_snapshot = {
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "step": step
  }

  torch.save(model_snapshot,
             os.path.join(config.model_save, "model_snapshot_{}.pth".format(tag)
                          if tag is not None else
                          "model_snapshot_{:07d}.pth".format(step)))

def snapshot_innernet(model, optimizer, config, step):
  model_snapshot = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "step": step
  }

  torch.save(model_snapshot,
             os.path.join(config.cell_save, "inner_net_{:03d}.pth".format(step)))



def load_model(model, file_name, optimizer=None):
  model_snapshot = torch.load(file_name, map_location=torch.device('cpu'))
  model.load_state_dict(model_snapshot["model"], strict=True)

  if optimizer is not None:
    optimizer.load_state_dict(model_snapshot["optimizer"])



def save_outphase(model, config, out):
    if config.use_gpu:
        model_dict = model.module.inner_net.state_dict()
    else:
        model_dict = model.inner_net.state_dict()

    for key in list(model_dict.keys()):
        model_dict[key.replace('0', 'inner_net')] = model_dict.pop(key)

    model_inner = InnerNet(config)
    model_inner.load_state_dict(model_dict, strict=True)
    model_inner.eval()

    out_phase = model_inner.inner_net(out)

    return out_phase.data.cpu().numpy()

class EarlyStopper(object):
  """ 
    Check whether the early stop condition (always 
    observing decrease in a window of time steps) is met.

    Usage:
      my_stopper = EarlyStopper([0, 0], 1)
      is_stop = my_stopper.tick([-1,-1]) # returns True
  """

  def __init__(self, init_val, win_size=10, is_decrease=True):
    if not isinstance(init_val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    self._win_size = win_size
    self._num_val = len(init_val)
    self._val = [[False] * win_size for _ in range(self._num_val)]
    self._last_val = init_val[:]
    self._comp_func = (lambda x, y: x < y) if is_decrease else (lambda x, y: x >= y)

  def tick(self, val):
    if not isinstance(val, list):
      raise ValueError("EarlyStopper only takes list of int/floats")

    assert len(val) == self._num_val

    for ii in range(self._num_val):
      self._val[ii].pop(0)

      if self._comp_func(val[ii], self._last_val[ii]):
        self._val[ii].append(True)
      else:
        self._val[ii].append(False)

      self._last_val[ii] = val[ii]

    is_stop = all([all(xx) for xx in self._val])

    return is_stop