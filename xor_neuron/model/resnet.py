import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if activation == 'ReLU':
            self.activation_fnc = nn.ReLU()
        elif activation == 'ELU':
            self.activation_fnc = nn.ELU()
        elif activation == 'LeakyReLU':
            self.activation_fnc = nn.LeakyReLU()
        elif activation == 'PReLU':
            self.activation_fnc = nn.PReLU()
        elif activation == 'SELU':
            self.activation_fnc = nn.SELU()
        elif activation == 'CELU':
            self.activation_fnc = nn.CELU()
        elif activation == 'GELU':
            self.activation_fnc = nn.GELU()
        elif activation == 'SiLU':
            self.activation_fnc = nn.SiLU()
        else:
            self.activation_fnc = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.activation_fnc(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation_fnc(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, config, num_classes=10):
        super(ResNet, self).__init__()
        self.config = config
        self.in_planes = 16
        self.num_blocks = config.model.num_blocks

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, self.num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.config.model.activation_fnc, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, labels):
        out = self.activation_fnc(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        loss = self.loss_func(out, labels)

        return out, loss


def resnet20(config):
    return ResNet(BasicBlock, config)
