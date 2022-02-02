import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .xorneuron import *
from .resnet_xor import BasicBlock_InnerNet, ResNet_Xor
from .resnet import ResNet

__all__ = ['Control_MLP_For_FoolBox', 'ComplexNeuronMLP_For_FoolBox',
           'ComplexNeuronConv_For_FoolBox', 'Control_Conv_For_FoolBox',
           'Xor_Resnet_For_FoolBox', 'ResNet20_Xor_adv_attack', 'ResNet20_adv_attack']

class Control_MLP_For_FoolBox(Control_MLP):
    def __init__(self, config):
        super(Control_MLP_For_FoolBox, self).__init__(config)

    def forward(self, x):
        x = x.reshape(-1, np.array(x.shape[1:]).prod())

        for i, fc in enumerate(self.model):
            if i == 0:
                out = fc(x)
            else:
                out = fc(out)
            out = F.relu(nn.LayerNorm(out.size()[1:], elementwise_affine=False)(out))
            out = self.drop_layer(out)

        out = self.fc_out(out)

        return out

class ComplexNeuronMLP_For_FoolBox(ComplexNeuronMLP):
    def __init__(self, config):
        super(ComplexNeuronMLP_For_FoolBox, self).__init__(config)

    def forward(self, x):
        x = x.reshape(-1, np.array(x.shape[1:]).prod())

        for i, fc in enumerate(self.outer_net):
            out, _ = fc(out) if i > 0 else fc(x)
            out = self.drop_layer(out)

        out = self.fc_out(out)

        return out

class ComplexNeuronConv_For_FoolBox(ComplexNeuronConv):
    def __init__(self, config):
        super(ComplexNeuronConv_For_FoolBox, self).__init__(config)

    def forward(self, x):
        batch_size = x.shape[0]

        for i, conv in enumerate(self.outer_net[:-1]):
            out, _ = conv(out) if i > 0 else conv(x)
            out = self.max_pool(out)
            out = self.drop_layer(out)

        # Effective flattening via 1x1 conv
        out, _ = self.outer_net[-1](out.view(batch_size, -1, 1, 1))
        out = self.drop_layer(out)
        # Output Layer
        out = self.fc_out(out.view(batch_size, -1))

        return out

class Control_Conv_For_FoolBox(Control_Conv):
    def __init__(self, config):
        super(Control_Conv_For_FoolBox, self).__init__(config)

        self.config = config

    def forward(self, x):
        batch_size = x.shape[0]

        for i, conv in enumerate(self.control_conv[:-1]):
            if i == 0:
                out = conv(x)
            else:
                out = conv(out)

            out = F.relu(nn.LayerNorm(out.size()[1:], elementwise_affine=False)(out))
            out = self.max_pool(out)
            out = self.drop_layer(out)

        out = self.control_conv[-1](out.view(batch_size, -1, 1, 1))
        out = self.drop_layer(out)

        out = self.fc_out(out.view(batch_size, -1))

        return out

class Xor_Resnet_For_FoolBox(ResNet_Xor):
    def __init__(self, block, config):
        super(Xor_Resnet_For_FoolBox, self).__init__(block, config, num_classes=10)

        self.config = config

    def forward(self, x):
        out = self._inner_net_forward(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out

class Resnet20_For_FoolBox(ResNet):
    def __init__(self, block, config):
        super(Resnet20_For_FoolBox, self).__init__(block, config, num_classes=10)

        self.config = config


    def forward(self, x, labels):
        out = self.activation_fnc(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet20_Xor_adv_attack(config):
    return Xor_Resnet_For_FoolBox(BasicBlock_InnerNet, config)

def ResNet20_adv_attack(config):
    return Resnet20_For_FoolBox(BasicBlock, config)