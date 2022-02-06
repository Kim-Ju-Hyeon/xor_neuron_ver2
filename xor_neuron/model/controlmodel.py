import numpy as np
import torch
import torch.nn as nn


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