from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import rand, matmul, diag, einsum

from .denselayer import DenseLayerWithComplexNeurons, DenseLayerWithQuadraticFuction
from .conv2dlayer import Conv2dLayerWithComplexNeurons, Conv2dLayerWithQuadraticFuction
from .rnncell import RNNCellWithComplexNeurons

EPS = float(np.finfo(np.float32).eps)

__all__ = ['InnerNet', 'InnerNet_V2', 'MultipleInnerNet', 'Control_MLP', 'Control_Conv', 'ComplexNeuronMLP', 'ComplexNeuronConv',
           'ComplexNeuronRNN', 'XorNeuronMLP', 'XorNeuronConv',
           'XorNeuronMLP_test', 'XorNeuronConv_test', 'QuadraticNeuronConv', 'QuadraticInnerNet',
           'QuadraticConvInnerNet', 'QuadraticLinearInnerNet', 'QuadraticNeuronMLP']


class InnerNet(nn.Module):
    def __init__(self, config):
        super(InnerNet, self).__init__()
        self.config = config
        self.arg_in_dim = config.model.arg_in_dim

        # inner net
        if self.config.model.inner_net == 'mlp':
            self.in_hidden_dim = config.model.in_hidden_dim

            self.inner_net = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(self.in_hidden_dim, 1))
            ]))
        elif self.config.model.inner_net == 'conv':
            self.in_channel = config.model.in_channel

            self.inner_net = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
                ('relu2', nn.ReLU()),
                ('conv3', nn.Conv2d(self.in_channel, 1, 1))
            ]))


        else:
            raise ValueError("Non-supported InnerNet!")

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
        if self.config.model.inner_net == 'mlp':
            out = self.inner_net(x)
        elif self.config.model.inner_net == 'conv':
            sqrt_batch_size = np.int(np.sqrt(x.shape[0]))
            assert sqrt_batch_size ** 2 == x.shape[0]
            out = x.T.reshape(1, self.arg_in_dim, sqrt_batch_size, sqrt_batch_size)
            out = self.inner_net(out)
            out = out.reshape(-1, 1)

        loss = self.loss_func(out, targets)
        return out, loss


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
        # batch_size = x.shape[0]
        # channel = x.shape[1]
        # input_size = x.shape[-1]
        #
        # inputs = x.reshape(batch_size, channel // self.arg_in_dim, -1, self.arg_in_dim)
        #
        # out = self.inner_net(inputs)
        #
        # out = out.reshape(batch_size, -1, input_size, input_size)
        out = self.inner_net(x)
        loss = self.loss_func(out, targets)

        return out, loss


class QuadraticInnerNet(nn.Module):
    def __init__(self):
        super(QuadraticInnerNet, self).__init__()
        self.quad = nn.Conv2d(6, 1, 1, bias=False)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        feature_map_size = inputs.shape[-1]
        x = torch.ones(batch_size, 6, feature_map_size, feature_map_size)
        x[:, 0, ...] = inputs[:, 0, ...] ** 2
        x[:, 1, ...] = inputs[:, 1, ...] ** 2
        x[:, 2, ...] = inputs[:, 0, ...] * inputs[:, 1, ...]
        x[:, 3, ...] = inputs[:, 0, ...]
        x[:, 4, ...] = inputs[:, 1, ...]
        x = x.cuda()
        x = self.quad(x)
        # x = F.relu(x)
        return x


class QuadraticConvInnerNet(nn.Module):
    def __init__(self, gpu=False):
        super(QuadraticConvInnerNet, self).__init__()
        self.gpu = gpu
        self.quad = nn.Conv2d(6, 1, 1, bias=False)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        feature_map_size = inputs.shape[-1]
        x = torch.ones(batch_size, 6, feature_map_size, feature_map_size)
        x[:, 0, ...] = inputs[:, 0, ...] ** 2
        x[:, 1, ...] = inputs[:, 1, ...] ** 2
        x[:, 2, ...] = inputs[:, 0, ...] * inputs[:, 1, ...]
        x[:, 3, ...] = inputs[:, 0, ...]
        x[:, 4, ...] = inputs[:, 1, ...]

        if self.gpu:
            x = x.cuda()
        x = self.quad(x)
        return x


class QuadraticLinearInnerNet(nn.Module):
    def __init__(self, gpu=False):
        super(QuadraticLinearInnerNet, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.b = nn.Linear(2, 1, bias=False)
        self.c = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]
        x_123 = self.A(x)
        x_45 = self.b(x)
        x_6 = self.c(torch.ones(batch_size, 1).cuda())

        out = einsum('ij, ij -> i', x_123, x_123).unsqueeze(dim=-1)
        out = einsum('ij, ij, ij -> i', out, x_45, x_6)

        return out.unsqueeze(dim=-1)


class MultipleInnerNet(nn.Module):
    def __init__(self, config):
        super(MultipleInnerNet, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.arg_in_dim = config.model.arg_in_dim

        # inner net
        if self.config.model.inner_net == 'mlp':
            self.in_hidden_dim = config.model.in_hidden_dim
            self.inner_net = nn.ModuleList()
            for i in range(self.num_cell_types):
                self.inner_net.append(
                    nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                        ('relu1', nn.ReLU()),
                        ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                        ('relu2', nn.ReLU()),
                        ('fc3', nn.Linear(self.in_hidden_dim, 1))
                    ]))
                )
        elif self.config.model.inner_net == 'conv':
            self.in_channel = config.model.in_channel
            self.inner_net = nn.ModuleList()
            for i in range(self.num_cell_types):
                self.inner_net.append(
                    nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
                        ('relu1', nn.ReLU()),
                        ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
                        ('relu2', nn.ReLU()),
                        ('conv3', nn.Conv2d(self.in_channel, 1, 1))
                    ]))
                )
        else:
            raise ValueError("Non-supported InnerNet!")


class Control_MLP(nn.Module):

    def __init__(self, config):
        super(Control_MLP, self).__init__()
        self.control = config
        self.input_dim = config.model.input_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout
        self.hidden_dim = config.model.hidden_dim

        self.loss = config.model.loss

        self.model = nn.ModuleList()

        for i in range(len(self.hidden_dim)):
            if i == 0:
                self.model.append(
                    nn.Linear(self.input_dim, self.hidden_dim[i])
                )

            else:
                self.model.append(
                    nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i])
                )

        self.fc_out = nn.Linear(self.hidden_dim[-1], self.num_classes)

        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        if config.model.activation_fnc == 'ReLU':
            self.activation_fnc = nn.ReLU()
        elif config.model.activation_fnc == 'ELU':
            self.activation_fnc = nn.ELU()
        elif config.model.activation_fnc == 'LeakyReLU':
            self.activation_fnc = nn.LeakyReLU()
        elif config.model.activation_fnc == 'PReLU':
            self.activation_fnc = nn.PReLU()
        elif config.model.activation_fnc == 'SELU':
            self.activation_fnc = nn.SELU()
        elif config.model.activation_fnc == 'CELU':
            self.activation_fnc = nn.CELU()
        elif config.model.activation_fnc == 'GELU':
            self.activation_fnc = nn.GELU()
        elif config.model.activation_fnc == 'SiLU':
            self.activation_fnc = nn.SiLU()
        else:
            self.activation_fnc = nn.ReLU()


    def forward(self, x, labels):
        x = x.reshape(-1, np.array(x.shape[1:]).prod())

        for i, fc in enumerate(self.model):
            if i == 0:
                out = fc(x)
            else:
                out = fc(out)
            out = self.activation_fnc(nn.LayerNorm(out.size()[1:], elementwise_affine=False)(out))
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss


class Control_Conv(nn.Module):
    def __init__(self, config):
        super(Control_Conv, self).__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.activation_fnc = config.model.activation_fnc
        self.num_classes = config.model.num_classes

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        if self.config.dataset.name == 'mnist':
            input_shape = [1, 1, 28, 28]
        elif self.config.dataset.name == 'cifar10' or 'cifar100':
            input_shape = [1, 3, 32, 32]
        else:
            raise ValueError("Non-supported dataset!")

        self.control_conv = nn.ModuleList()

        for i in range(len(self.channel)):
            if i == 0:
                self.control_conv.append(nn.Conv2d(self.input_channel,
                                                   self.channel[i],
                                                   self.kernel_size[i],
                                                   stride=self.stride[i],
                                                   padding=self.zero_pad[i]))

            elif i < len(self.channel) - 1:
                self.control_conv.append(nn.Conv2d(self.channel[i - 1],
                                                   self.channel[i],
                                                   self.kernel_size[i],
                                                   stride=self.stride[i],
                                                   padding=self.zero_pad[i]))

            else:
                for j in range(len(self.channel) - 1):
                    input_shape = self.control_conv[j](torch.rand(*input_shape))[0].data.shape
                    input_shape = self.max_pool(torch.rand(*input_shape)).data.shape
                    input_shape = [1, input_shape[0], input_shape[1], input_shape[2]]

                self.control_conv.append(nn.Conv2d(np.prod(list(input_shape)),
                                                   self.channel[i],
                                                   self.kernel_size[i],
                                                   stride=self.stride[i],
                                                   padding=self.zero_pad[i]))

        self.fc_out = nn.Linear(self.channel[-1], self.num_classes)
        self.drop_layer = nn.Dropout(p=0.5)
        self.loss_func = nn.CrossEntropyLoss()

        if config.model.activation_fnc == 'ReLU':
            self.activation_fnc = nn.ReLU()
        elif config.model.activation_fnc == 'ELU':
            self.activation_fnc = nn.ELU()
        elif config.model.activation_fnc == 'LeakyReLU':
            self.activation_fnc = nn.LeakyReLU()
        elif config.model.activation_fnc == 'PReLU':
            self.activation_fnc = nn.PReLU()
        elif config.model.activation_fnc == 'SELU':
            self.activation_fnc = nn.SELU()
        elif config.model.activation_fnc == 'CELU':
            self.activation_fnc = nn.CELU()
        elif config.model.activation_fnc == 'GELU':
            self.activation_fnc = nn.GELU()
        elif config.model.activation_fnc == 'SiLU':
            self.activation_fnc = nn.SiLU()
        else:
            self.activation_fnc = nn.ReLU()

    def forward(self, x, labels):
        batch_size = x.shape[0]

        for i, conv in enumerate(self.control_conv[:-1]):
            if i == 0:
                out = conv(x)
            else:
                out = conv(out)

            out = self.activation_fnc(nn.LayerNorm(out.size()[1:], elementwise_affine=False)(out))
            out = self.max_pool(out)
            out = self.drop_layer(out)

        out = self.control_conv[-1](out.view(batch_size, -1, 1, 1))
        out = self.drop_layer(out)

        out = self.fc_out(out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss


class QuadraticNeuronMLP(nn.Module):

    def __init__(self, config):
        super(QuadraticNeuronMLP, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout
        # self.save_all = config.to_save_cell

        # inner net
        self.inner_net = QuadraticLinearInnerNet()

        # outer net
        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            if i == 0:
                self.outer_net.append(
                    DenseLayerWithQuadraticFuction(self.inner_net,
                                                   self.arg_in_dim,
                                                   self.input_dim,
                                                   self.out_hidden_dim[i])
                )
            else:
                self.outer_net.append(
                    DenseLayerWithQuadraticFuction(self.inner_net,
                                                   self.arg_in_dim,
                                                   self.out_hidden_dim[i - 1],
                                                   self.out_hidden_dim[i])
                )

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1], self.num_classes)

        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm, nn.Linear):
                                nn.init.xavier_uniform_(mmm.weight.data)
                                if mmm.bias is not None:
                                    mmm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        x = x.reshape(-1, np.array(x.shape[1:]).prod())
        # in2cells_per_layer : batch_size x num_layers x num_cell_types x ... x arity
        in2cells_per_layer = []
        for i, fc in enumerate(self.outer_net):
            out, in2cells = fc(out, collect=collect) if i > 0 else fc(x, collect=collect)
            in2cells_per_layer.append(in2cells)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class ComplexNeuronMLP(nn.Module):

    def __init__(self, config):
        super(ComplexNeuronMLP, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout
        # self.save_all = config.to_save_cell

        # inner net
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(
                nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(self.in_hidden_dim, 1))
                ]))
            )

        # outer net
        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            if i == 0:
                self.outer_net.append(
                    DenseLayerWithComplexNeurons(self.inner_net,
                                                 self.arg_in_dim,
                                                 self.input_dim,
                                                 self.out_hidden_dim[i])
                )
            else:
                self.outer_net.append(
                    DenseLayerWithComplexNeurons(self.inner_net,
                                                 self.arg_in_dim,
                                                 self.out_hidden_dim[i - 1],
                                                 self.out_hidden_dim[i])
                )

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1], self.num_classes)

        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm, nn.Linear):
                                nn.init.xavier_uniform_(mmm.weight.data)
                                if mmm.bias is not None:
                                    mmm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        x = x.reshape(-1, np.array(x.shape[1:]).prod())
        # in2cells_per_layer : batch_size x num_layers x num_cell_types x ... x arity
        in2cells_per_layer = []
        for i, fc in enumerate(self.outer_net):
            out, in2cells = fc(out, collect=collect) if i > 0 else fc(x, collect=collect)
            in2cells_per_layer.append(in2cells)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class QuadraticNeuronConv(nn.Module):

    def __init__(self, config):
        super(QuadraticNeuronConv, self).__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = QuadraticConvInnerNet()

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # outer net
        if self.config.dataset.name == 'mnist':
            input_shape = [1, 1, 28, 28]
        elif self.config.dataset.name == 'cifar10' or 'cifar100':
            input_shape = [1, 3, 32, 32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_channel)):
            if i == 0:
                self.outer_net.append(
                    Conv2dLayerWithQuadraticFuction(self.inner_net,
                                                    self.arg_in_dim,
                                                    self.input_channel,
                                                    self.out_channel[i],
                                                    self.kernel_size[i],
                                                    stride=self.stride[i],
                                                    padding=self.zero_pad[i])
                )

            elif i < len(self.out_channel) - 1:
                self.outer_net.append(
                    Conv2dLayerWithQuadraticFuction(self.inner_net,
                                                    self.arg_in_dim,
                                                    self.out_channel[i - 1],
                                                    self.out_channel[i],
                                                    self.kernel_size[i],
                                                    stride=self.stride[i],
                                                    padding=self.zero_pad[i])
                )

            else:
                # calculate the expected shape of input to fc_out_1
                for j in range(len(self.out_channel) - 1):
                    input_shape = self.outer_net[j](torch.rand(*input_shape))[0].data.shape
                    input_shape = self.max_pool(torch.rand(*input_shape)).data.shape

                self.outer_net.append(
                    Conv2dLayerWithQuadraticFuction(self.inner_net,
                                                    self.arg_in_dim,
                                                    np.prod(list(input_shape)),
                                                    self.out_channel[i],
                                                    self.kernel_size[i],
                                                    stride=self.stride[i],
                                                    padding=self.zero_pad[i])
                )

        # output layer
        self.fc_out = nn.Linear(self.out_channel[-1], self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        # self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm, nn.Conv2d):
                                nn.init.kaiming_uniform_(mmm.weight.data, a=math.sqrt(5))
                                if mmm.bias is not None:
                                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                                    bound = 1 / math.sqrt(fan_in)
                                    nn.init.uniform_(mmm.bias.data, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        # in2cells_per_layer : num_layers x [batch_size x num_cell_types x ... x arity]
        in2cells_per_layer = []

        if len(self.outer_net) == 1:
            out, in2cells = self.outer_net[0](x, collect=collect)

            out = self.max_pool(out)
            out = self.drop_layer(out)
        else:
            for i, conv in enumerate(self.outer_net[:-1]):
                out, in2cells = conv(out, collect=collect) if i > 0 else conv(x, collect=collect)
                if i == 2:  # collect 2nd layer only
                    in2cells_per_layer.append(in2cells)
                out = self.max_pool(out)
                out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out, _ = self.outer_net[-1](out.view(batch_size, -1, 1, 1), collect=collect)
        out = self.drop_layer(out)
        # Output Layer
        out = self.fc_out(out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class ComplexNeuronConv(nn.Module):

    def __init__(self, config):
        super(ComplexNeuronConv, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(
                nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
                    ('relu2', nn.ReLU()),
                    ('conv3', nn.Conv2d(self.in_channel, 1, 1))
                ]))
            )

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # outer net
        if self.config.dataset.name == 'mnist':
            input_shape = [1, 1, 28, 28]
        elif self.config.dataset.name == 'cifar10' or 'cifar100':
            input_shape = [1, 3, 32, 32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_channel)):
            if i == 0:
                self.outer_net.append(
                    Conv2dLayerWithComplexNeurons(self.inner_net,
                                                  self.arg_in_dim,
                                                  self.input_channel,
                                                  self.out_channel[i],
                                                  self.kernel_size[i],
                                                  stride=self.stride[i],
                                                  padding=self.zero_pad[i])
                )

            elif i < len(self.out_channel) - 1:
                self.outer_net.append(
                    Conv2dLayerWithComplexNeurons(self.inner_net,
                                                  self.arg_in_dim,
                                                  self.out_channel[i - 1],
                                                  self.out_channel[i],
                                                  self.kernel_size[i],
                                                  stride=self.stride[i],
                                                  padding=self.zero_pad[i])
                )

            else:
                # calculate the expected shape of input to fc_out_1
                for j in range(len(self.out_channel) - 1):
                    input_shape = self.outer_net[j](torch.rand(*input_shape))[0].data.shape
                    input_shape = self.max_pool(torch.rand(*input_shape)).data.shape

                self.outer_net.append(
                    Conv2dLayerWithComplexNeurons(self.inner_net,
                                                  self.arg_in_dim,
                                                  np.prod(list(input_shape)),
                                                  self.out_channel[i],
                                                  self.kernel_size[i],
                                                  stride=self.stride[i],
                                                  padding=self.zero_pad[i])
                )

        # output layer
        self.fc_out = nn.Linear(self.out_channel[-1], self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        # self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Sequential):
                        for mmm in mm:
                            if isinstance(mmm, nn.Conv2d):
                                nn.init.kaiming_uniform_(mmm.weight.data, a=math.sqrt(5))
                                if mmm.bias is not None:
                                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                                    bound = 1 / math.sqrt(fan_in)
                                    nn.init.uniform_(mmm.bias.data, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels, collect=False):
        batch_size = x.shape[0]
        # in2cells_per_layer : num_layers x [batch_size x num_cell_types x ... x arity]
        in2cells_per_layer = []

        if len(self.outer_net) == 1:
            out, in2cells = self.outer_net[0](x, collect=collect)

            out = self.max_pool(out)
            out = self.drop_layer(out)
        else:
            for i, conv in enumerate(self.outer_net[:-1]):
                out, in2cells = conv(out, collect=collect) if i > 0 else conv(x, collect=collect)
                if i == 2:  # collect 2nd layer only
                    in2cells_per_layer.append(in2cells)
                out = self.max_pool(out)
                out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out, _ = self.outer_net[-1](out.view(batch_size, -1, 1, 1), collect=collect)
        out = self.drop_layer(out)
        # Output Layer
        out = self.fc_out(out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss, in2cells_per_layer


class ComplexNeuronRNN(nn.Module):

    def __init__(self, config, ntoken):
        super(ComplexNeuronRNN, self).__init__()
        self.config = config
        self.num_cell_types = config.model.num_cell_types
        self.embedding_dim = config.model.embedding_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim

        self.dropout = nn.Dropout(config.model.dropout)
        self.encoder = nn.Embedding(ntoken, self.embedding_dim)  # Token2Embeddings
        # inner net
        self.inner_net = nn.ModuleList()
        for i in range(self.num_cell_types):
            self.inner_net.append(
                nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(self.in_hidden_dim, 1))
                ]))
            )

        # outer net
        self.outer_net = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            if i == 0:
                self.outer_net.append(
                    RNNCellWithComplexNeurons(self.inner_net,
                                              self.arg_in_dim,
                                              self.embedding_dim,
                                              self.out_hidden_dim[i])
                )
            else:
                self.outer_net.append(
                    RNNCellWithComplexNeurons(self.inner_net,
                                              self.arg_in_dim,
                                              self.out_hidden_dim[i - 1],
                                              self.out_hidden_dim[i])
                )
        self.decoder = nn.Linear(self.out_hidden_dim[-1], ntoken)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        initrange = 0.05
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, labels, mask=None, hx=None):
        # input : seq_len, batch_size
        # labels : seq_len * batch_size
        # emb : seq_len, batch_size, emb_dim
        # hx : num_layer * [batch_size, hidden_dim]
        emb = self.dropout(self.encoder(x))
        if not hx:
            hx = [torch.zeros(x.size(1), self.out_hidden_dim[i], device=x.device) for i in
                  range(len(self.out_hidden_dim))]
        if not mask:
            mask = []
            for i in range(len(self.out_hidden_dim)):
                mask.append(torch.ones(self.out_hidden_dim[i], device=x.device))

        # in2cells_per_layer : batch_size x num_layers x num_cell_types x ... x arity
        in2cells_per_layer = []
        output = []
        for i_seq in range(emb.size(0)):
            for i_layer, rnn_layer in enumerate(self.outer_net):
                if i_layer == 0 and i_seq == 0:
                    hx_update, in2cells = rnn_layer(emb[i_seq])
                elif i_layer > 0 and i_seq == 0:
                    hx_update, in2cells = rnn_layer(hx[i_layer - 1])
                elif i_layer == 0 and i_seq > 0:
                    hx_update, in2cells = rnn_layer(emb[i_seq], hx[i_layer])
                else:
                    hx_update, in2cells = rnn_layer(hx[i_layer - 1], hx[i_layer])

                hx[i_layer] = hx_update * mask[i_layer]
                if i_seq == emb.size(0) - 1:
                    in2cells_per_layer.append(in2cells)
            out = self.decoder(hx[-1])
            output.append(out)

        # output : seq_len * batch_size, ntoken
        output = torch.cat(output, 0)
        # labels : seq_len * batch_size
        loss = self.loss_func(output, labels)

        return output, hx, loss, in2cells_per_layer


class XorNeuronMLP(nn.Module):

    def __init__(self, config):
        super(XorNeuronMLP, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.in_hidden_dim, 1))
        ]))

        # outer net
        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            self.layer_norm.append(nn.LayerNorm(self.out_hidden_dim[i], elementwise_affine=False))
            if i == 0:
                self.outer_net.append(nn.Linear(self.input_dim, self.out_hidden_dim[i]))
            else:
                self.outer_net.append(nn.Linear(self.out_hidden_dim[i] // self.arg_in_dim, self.out_hidden_dim[i]))

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1] // self.arg_in_dim, self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels):
        batch_size = x.shape[0]
        x = x.reshape(-1, np.array(x.shape[1:]).prod())

        for i, fc in enumerate(self.outer_net):
            out = fc(out) if i > 0 else fc(x)
            out = self.layer_norm[i](out)
            out = self.inner_net(out.reshape(batch_size, -1, self.arg_in_dim)).reshape(batch_size, -1)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss


class XorNeuronConv(nn.Module):

    def __init__(self, config):
        super(XorNeuronConv, self).__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(self.in_channel, 1, 1))
        ]))

        # outer net
        if self.config.dataset.name == 'mnist':
            x_shape = [28]
        elif self.config.dataset.name == 'cifar10':
            x_shape = [32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_channel)):
            x_shape.append(((x_shape[-1] - self.kernel_size[i] + 2 * self.zero_pad[i]) // self.stride[i] + 1))
            self.layer_norm.append(
                nn.LayerNorm([self.out_channel[i], x_shape[-1], x_shape[-1]], elementwise_affine=False))
            x_shape.append(x_shape[-1] // 2)
            if i == 0:
                self.outer_net.append(nn.Conv2d(in_channels=self.input_channel,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))
            else:
                self.outer_net.append(nn.Conv2d(in_channels=self.out_channel[i - 1] // self.arg_in_dim,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # output layer
        self.fc_out = nn.ModuleList()
        # self.fc_out.append(nn.Linear(self.out_channel[-1] // self.arg_in_dim + x_shape ** 2, 256))
        self.fc_out.append(nn.Conv2d(in_channels=self.out_channel[-1] // self.arg_in_dim * x_shape[-1] * x_shape[-1],
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1))
        self.fc_out.append(nn.Linear(128, self.num_classes))

        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels):
        batch_size = x.shape[0]

        for i, conv in enumerate(self.outer_net):
            # ConvLayer
            out = conv(out) if i > 0 else conv(x)
            out = self.layer_norm[i](out)
            # InnerNet
            out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
            out = self.inner_net(out)
            out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
            # MaxPooling
            out = self.max_pool(out)
            out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out = self.fc_out[0](out.view(batch_size, -1, 1, 1))
        # InnerNet
        out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
        out = self.inner_net(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
        # OutputLayer
        out = self.fc_out[1](out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss


class XorNeuronMLP_test(nn.Module):

    def __init__(self, config):
        super(XorNeuronMLP_test, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_hidden_dim
        self.out_hidden_dim = config.model.out_hidden_dim
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.in_hidden_dim, 1))
        ]))

        # outer net
        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_hidden_dim)):
            self.layer_norm.append(nn.LayerNorm(self.out_hidden_dim[i], elementwise_affine=False))
            if i == 0:
                self.outer_net.append(nn.Linear(self.input_dim, self.out_hidden_dim[i]))
            else:
                self.outer_net.append(nn.Linear(self.out_hidden_dim[i] // self.arg_in_dim, self.out_hidden_dim[i]))

        # output layer
        self.fc_out = nn.Linear(self.out_hidden_dim[-1] // self.arg_in_dim, self.num_classes)
        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels):
        batch_size = x.shape[0]
        x = x.reshape(-1, np.array(x.shape[1:]).prod())
        input2inner = {}

        for i, fc in enumerate(self.outer_net):
            out = fc(out) if i > 0 else fc(x)
            out = self.layer_norm[i](out)
            # Collect Inputs to InnerNet
            input2inner[i] = out.data.cpu().numpy()
            out = self.inner_net(out.reshape(batch_size, -1, self.arg_in_dim)).reshape(batch_size, -1)
            out = self.drop_layer(out)

        out = self.fc_out(out)
        loss = self.loss_func(out, labels)

        return out, loss, input2inner


class XorNeuronConv_test(nn.Module):

    def __init__(self, config):
        super(XorNeuronConv_test, self).__init__()
        self.config = config
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.out_channel = config.model.out_channel
        self.kernel_size = config.model.kernel_size
        self.zero_pad = config.model.zero_pad
        self.stride = config.model.stride
        self.num_classes = config.model.num_classes
        self.dropout = config.model.dropout

        # inner net
        self.inner_net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.arg_in_dim, self.in_channel, 1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(self.in_channel, self.in_channel, 1)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv2d(self.in_channel, 1, 1))
        ]))

        # outer net
        if self.config.dataset.name == 'mnist':
            x_shape = [28]
        elif self.config.dataset.name == 'cifar10':
            x_shape = [32]
        else:
            raise ValueError("Non-supported dataset!")

        self.outer_net = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        for i in range(len(self.out_channel)):
            x_shape.append(((x_shape[-1] - self.kernel_size[i] + 2 * self.zero_pad[i]) // self.stride[i] + 1))
            self.layer_norm.append(
                nn.LayerNorm([self.out_channel[i], x_shape[-1], x_shape[-1]], elementwise_affine=False))
            x_shape.append(x_shape[-1] // 2)
            if i == 0:
                self.outer_net.append(nn.Conv2d(in_channels=self.input_channel,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))
            else:
                self.outer_net.append(nn.Conv2d(in_channels=self.out_channel[i - 1] // self.arg_in_dim,
                                                out_channels=self.out_channel[i],
                                                kernel_size=self.kernel_size[i],
                                                stride=self.stride[i],
                                                padding=self.zero_pad[i]))

        # max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # output layer
        self.fc_out = nn.ModuleList()
        # self.fc_out.append(nn.Linear(self.out_channel[-1] // self.arg_in_dim + x_shape ** 2, 256))
        self.fc_out.append(nn.Conv2d(in_channels=self.out_channel[-1] // self.arg_in_dim * x_shape[-1] * x_shape[-1],
                                     out_channels=256,
                                     kernel_size=1,
                                     stride=1))
        self.fc_out.append(nn.Linear(128, self.num_classes))

        # dropout layer
        self.drop_layer = nn.Dropout(p=self.dropout)

        if self.config.model.loss == 'CrossEntropy':
            self.loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Non-supported loss function!")

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.fc_out, self.inner_net, self.outer_net] if xx is not None
        ]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential) or isinstance(m, nn.ModuleList):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, labels):
        batch_size = x.shape[0]
        input2inner = {}
        for i, conv in enumerate(self.outer_net):
            # ConvLayer
            out = conv(out) if i > 0 else conv(x)
            out = self.layer_norm[i](out)
            out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
            # Collect Inputs to InnerNet
            input2inner[i] = np.moveaxis(out.data.cpu().numpy(), 1, -1).reshape(-1, self.arg_in_dim)
            # InnerNet
            out = self.inner_net(out)
            out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
            # MaxPooling
            out = self.max_pool(out)
            out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out = self.fc_out[0](out.view(batch_size, -1, 1, 1))
        # InnerNet
        out = out.reshape(-1, self.arg_in_dim, out.shape[2], out.shape[3])
        # Collect Inputs to InnerNet
        input2inner[i + 1] = np.moveaxis(out.data.cpu().numpy(), 1, -1).reshape(-1, self.arg_in_dim)
        out = self.inner_net(out)
        out = out.reshape(batch_size, out.shape[0] // batch_size, out.shape[-2], out.shape[-1])
        # OutputLayer
        out = self.fc_out[1](out.view(batch_size, -1))
        loss = self.loss_func(out, labels)

        return out, loss, input2inner
