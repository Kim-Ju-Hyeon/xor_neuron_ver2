import math
import numpy as np
import warnings

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd
from utils.conv_utils import _pair
from utils.common_types import _size_2_t


class Conv2dLayerWithComplexNeurons(_ConvNd):
    def __init__(
        self,
        cell_types,
        arity: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dLayerWithComplexNeurons, self).__init__(
            in_channels, arity * out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = len(cell_types)
        self.out_channels = out_channels

        assert out_channels % self.num_cell_types == 0
        self.cell_indices = np.repeat(range(self.num_cell_types), self.out_channels // self.num_cell_types)


    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, collect=False) -> Tensor:
        all_inputs = self._conv_forward(input, self.weight)
        all_inputs = torch.nn.LayerNorm([all_inputs.shape[1], all_inputs.shape[2], all_inputs.shape[3]],
                                        elementwise_affine=False)(all_inputs)

        all_outputs = []
        for i in range(self.out_channels):
            o = self.cell_types[self.cell_indices[i]](all_inputs[:, i*self.arity:(i+1)*self.arity, ...])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, 1)

        # Input to multiple cells
        if collect:
            in2cells = all_inputs.data.cpu().numpy().reshape(
                        all_inputs.shape[0],
                        all_inputs.shape[1],
                        -1)
            in2cells = in2cells.reshape(
                        in2cells.shape[0],
                        self.num_cell_types,
                        self.out_channels // self.num_cell_types,
                        self.arity,
                        in2cells.shape[-1])
            in2cells = np.swapaxes(in2cells, -2, -1)
            in2cells = in2cells.reshape(
                        in2cells.shape[0],
                        self.num_cell_types,
                        -1,
                        self.arity)
        else:
            in2cells = None


        return all_outputs, in2cells

class Conv2dLayerWithComplexNeuronsForResnet(_ConvNd):
    def __init__(
            self,
            cell_types,
            arity: int,
            in_channels: int,
            out_channels: int,
            kernel_size: _size_2_t,
            stride: _size_2_t = 1,
            padding: _size_2_t = 0,
            dilation: _size_2_t = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dLayerWithComplexNeuronsForResnet, self).__init__(
            in_channels, arity*out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = len(cell_types)
        self.out_channels = out_channels

        assert out_channels % self.num_cell_types == 0
        self.cell_indices = np.repeat(range(self.num_cell_types), self.out_channels // self.num_cell_types)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, skip_connection=None, collect=False) -> Tensor:
        # print(input.shape)
        all_inputs = self._conv_forward(input, self.weight)
        # print(f'ff: {all_inputs.shape}')

        if skip_connection is not None:
            # print(f"skip_connection shape: {skip_connection.shape}")
            all_inputs += skip_connection

        all_outputs = []
        for i in range(self.out_channels):
            o = self.cell_types[self.cell_indices[i]](all_inputs[:, i * self.arity:(i + 1) * self.arity, ...])
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, 1)
        # print(f'after innernet: {all_outputs.shape}')

        # Input to multiple cells
        if collect:
            in2cells = all_inputs.data.cpu().numpy().reshape(
                all_inputs.shape[0],
                all_inputs.shape[1],
                -1)
            in2cells = in2cells.reshape(
                in2cells.shape[0],
                self.num_cell_types,
                self.out_channels // self.num_cell_types,
                self.arity,
                in2cells.shape[-1])
            in2cells = np.swapaxes(in2cells, -2, -1)
            in2cells = in2cells.reshape(
                in2cells.shape[0],
                self.num_cell_types,
                -1,
                self.arity)
        else:
            in2cells = None

        return all_outputs


class Conv2dLayerWithQuadraticFuction(_ConvNd):
    def __init__(
        self,
        cell_types,
        arity: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dLayerWithQuadraticFuction, self).__init__(
            in_channels, arity * out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.cell_types = cell_types
        self.arity = arity
        self.num_cell_types = 2
        self.out_channels = out_channels

        assert out_channels % self.num_cell_types == 0
        # self.cell_indices = np.repeat(range(self.num_cell_types), self.out_channels // self.num_cell_types)
        # print(self.cell_indices.shape)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor, collect=False) -> Tensor:
        print(f'Innernet reshaping input: {input.shape}')
        all_inputs = self._conv_forward(input, self.weight)
        # all_inputs = torch.nn.LayerNorm([all_inputs.shape[1], all_inputs.shape[2], all_inputs.shape[3]],
        #                                 elementwise_affine=False)(all_inputs)
        print(f'Innernet Input: {all_inputs.shape} ')

        all_outputs = []
        for i in range(self.out_channels):
            part_inputs = all_inputs[:, i*self.arity:(i+1)*self.arity, ...]
            o = self.cell_types(part_inputs)
            all_outputs.append(o)
        all_outputs = torch.cat(all_outputs, 1)

        return all_outputs, None


