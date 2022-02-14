import os
import yaml
import time
import argparse
from easydict import EasyDict as edict
import numpy as np


def get_config(config_file, exp_dir=None):
      """ Construct and snapshot hyper parameters """
      config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

      # create hyper parameters
      config.run_id = str(os.getpid())

      if config.exp_name == 'Test':
          config.exp_name = '_'.join([config.exp_name, config.model.inner_net,
                                      config.dataset.name, time.strftime('%H%M%S')])
      else:
          config.exp_name = '_'.join([
              config.model.name, str(config.exp_name), config.dataset.name,
              time.strftime('%H%M%S')
          ])

      if exp_dir is not None:
        config.exp_dir = exp_dir


      config.save_dir = os.path.join(config.exp_dir, config.exp_name)
      config.model_save = os.path.join(config.save_dir, "model_save")

      # snapshot hyperparameters
      mkdir(config.exp_dir)
      mkdir(config.save_dir)
      mkdir(config.model_save)

      save_name = os.path.join(config.save_dir, 'config.yaml')
      yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

      # config.seed = int(str(config.seed) + sample_id)

      return config

def get_config_for_multi_test(config_file, sample_id, exp_num):
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

    config.exp_name = exp_num
    config.seed = np.random.randint(10000)

    config.exp_name = '_'.join([
        config.model.name, str(config.exp_name), config.dataset.name,
        time.strftime('%H%M%S')
    ])

    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    config.model_save = os.path.join(config.save_dir, "model_save")

    mkdir(config.exp_dir)
    mkdir(config.save_dir)
    mkdir(config.model_save)

    save_name = os.path.join(config.save_dir, 'config.yaml')

    config.seed = int(str(config.seed) + str(sample_id))
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    return config

def get_config_for_multi_act_exp(config_file, sample_id, exp_num, act_fnc):
    config = edict(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))

    config.exp_name = exp_num
    config.seed = np.random.randint(10000)
    config.model.activation_fnc = act_fnc
    config.exp_dir = os.path.join(config.exp_dir, act_fnc)

    config.exp_name = '_'.join([
        config.model.name, str(config.exp_name), config.dataset.name,
        time.strftime('%H%M%S')
    ])

    config.save_dir = os.path.join(config.exp_dir, config.exp_name)
    config.model_save = os.path.join(config.save_dir, "model_save")

    mkdir(config.exp_dir)
    mkdir(config.save_dir)
    mkdir(config.model_save)

    save_name = os.path.join(config.save_dir, 'config.yaml')

    config.seed = int(str(config.seed) + str(sample_id))
    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

    return config

def edict2dict(edict_obj):
      dict_obj = {}

      for key, vals in edict_obj.items():
        if isinstance(vals, edict):
          dict_obj[key] = edict2dict(vals)
        else:
          dict_obj[key] = vals

      return dict_obj


def mkdir(folder):
      if not os.path.isdir(folder):
        os.makedirs(folder)
