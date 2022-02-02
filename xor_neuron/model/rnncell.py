import math
import numpy as np
import torch
from torch import Tensor
from typing import Optional
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter


class RNNCellWithComplexNeurons(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, cell_types, arity: int, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super(RNNCellWithComplexNeurons, self).__init__()
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = len(cell_types)
        assert hidden_size % self.num_cell_types == 0
        self.cell_indices = np.repeat(range(self.num_cell_types), hidden_size // self.num_cell_types)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(arity * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(arity * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(arity * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(arity * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')

        all_inputs = F.linear(input, self.weight_ih, self.bias_ih)\
                   + F.linear(hx, self.weight_hh, self.bias_hh)
        all_inputs = torch.nn.LayerNorm(len(self.bias_hh), elementwise_affine=False)(all_inputs)
        all_outputs = []
        for i in range(self.hidden_size):
            o = self.cell_types[self.cell_indices[i]](all_inputs[..., i * self.arity:(i + 1) * self.arity])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, -1)
        # Input to multiple cells
        in2cells = all_inputs.data.cpu().numpy().reshape(
            -1,
            self.num_cell_types,
            self.hidden_size // self.num_cell_types,
            self.arity)

        return all_outputs, in2cells


    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        # emb size(bptt, bsz, embsize)
        # hid size(bsz, nhid)
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)