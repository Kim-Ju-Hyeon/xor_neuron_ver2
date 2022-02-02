import math
import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module


class DenseLayerWithComplexNeurons(Module):
    __constants__ = ['arity', 'in_features', 'out_features']
    arity: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,  cell_types, arity: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(DenseLayerWithComplexNeurons, self).__init__()
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = len(cell_types)
        self.in_features = in_features
        self.out_features = out_features

        assert self.out_features % self.num_cell_types == 0  # make sure each cell type has the same # of neurons
        self.cell_indices = np.repeat(range(self.num_cell_types), self.out_features // self.num_cell_types)

        self.weight = Parameter(torch.Tensor(arity * out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(arity * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, collect=False) -> Tensor:
        all_inputs = F.linear(input, self.weight, self.bias)
        all_inputs = torch.nn.LayerNorm(len(self.bias), elementwise_affine=False)(all_inputs)
        # all_inputs = torch.nn.LayerNorm(all_inputs.size()[1:])(all_inputs)
        all_outputs = []

        for i in range(self.out_features):
            o = self.cell_types[self.cell_indices[i]](all_inputs[..., i*self.arity:(i+1)*self.arity])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, -1)

        # Input to multiple cells
        if collect:
            in2cells = all_inputs.data.cpu().numpy().reshape(
                        -1,
                        self.num_cell_types,
                        self.out_features // self.num_cell_types,
                        self.arity)
        else:
            in2cells = None

        return all_outputs, in2cells

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features,
                                                                 self.bias is not None)


class DenseLayerWithQuadraticFuction(Module):
    __constants__ = ['arity', 'in_features', 'out_features']
    arity: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,  cell_types, arity: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(DenseLayerWithQuadraticFuction, self).__init__()
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = 2
        self.in_features = in_features
        self.out_features = out_features

        assert self.out_features % self.num_cell_types == 0  # make sure each cell type has the same # of neurons
        self.cell_indices = np.repeat(range(self.num_cell_types), self.out_features // self.num_cell_types)

        self.weight = Parameter(torch.Tensor(arity * out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(arity * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, collect=False) -> Tensor:
        all_inputs = F.linear(input, self.weight, self.bias)
        # all_inputs = torch.nn.LayerNorm(len(self.bias), elementwise_affine=False)(all_inputs)
        # all_inputs = torch.nn.LayerNorm(all_inputs.size()[1:])(all_inputs)
        all_outputs = []

        for i in range(self.out_features):
            o = self.cell_types(all_inputs[..., i*self.arity:(i+1)*self.arity])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, -1)

        # Input to multiple cells
        if collect:
            in2cells = all_inputs.data.cpu().numpy().reshape(
                        -1,
                        self.num_cell_types,
                        self.out_features // self.num_cell_types,
                        self.arity)
        else:
            in2cells = None

        return all_outputs, in2cells

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features,
                                                                 self.bias is not None)