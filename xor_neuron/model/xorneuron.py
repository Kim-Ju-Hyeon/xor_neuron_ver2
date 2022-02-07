from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
from torch import einsum


EPS = float(np.finfo(np.float32).eps)

__all__ = ['InnerNet_V2', 'QuadraticInnerNet']


class InnerNet_V2(nn.Module):
    def __init__(self, config):
        super(InnerNet_V2, self).__init__()
        self.config = config
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_channel

        self.inner_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.in_hidden_dim, 1))]))

        self.loss_func = nn.MSELoss()
        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, targets):
        out = self.inner_net(x)
        loss = self.loss_func(out, targets)

        return out, loss


class QuadraticInnerNet(nn.Module):
    def __init__(self):
        super(QuadraticInnerNet, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 1, bias=True)

    def forward(self, x):
        if len(x.shape) == 2:
            inputs = x.reshape(x.shape[0], -1, 2)

            x_123 = self.A(inputs)
            x_45 = self.b(inputs)

            out = einsum('ijk, ijk -> ij', x_123, inputs).unsqueeze(dim=-1)
            out += x_45
            out = out.squeeze()

        elif len(x.shape) == 4:
            # inputs = x.reshape(x.shape[0], x.shape[1] // 2, -1, 2)
            inputs = x

            x_123 = self.A(inputs)
            x_45 = self.b(inputs)

            out = einsum('ijkh, ijkh -> ijk', x_123, inputs).unsqueeze(dim=-1)
            out += x_45
            out = out.squeeze()
            # out = out.reshape(x.shape[0], -1, x.shape[-1], x.shape[-1])

        else:
            raise ValueError('Invalid input shape')

        return out