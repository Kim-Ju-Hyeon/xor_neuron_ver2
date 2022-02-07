import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.xorneuron import QuadraticInnerNet

from collections import OrderedDict

__all__ = ['ResNet20_Xor', 'BasicBlock_InnerNet']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_InnerNet(nn.Module):
    expansion = 1

    def __init__(self, inner_net, in_planes, planes, arg_in_dim, stride=1):
        super(BasicBlock_InnerNet, self).__init__()
        self.inner_net = inner_net
        self.arg_in_dim = arg_in_dim

        self.conv1 = nn.Conv2d(
            in_planes, planes*arg_in_dim, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes*arg_in_dim)

        self.conv2 = nn.Conv2d(
            planes, planes*arg_in_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes*arg_in_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(
                    x[:, :, ::2, ::2],
                    (0, 0, 0, 0, planes // 4, planes // 4),
                    "constant",
                    0,
                ))

    def _inner_net_forward(self, x):
        batch_size = x.shape[0]
        channel = x.shape[1]
        input_size = x.shape[-1]

        input = x.reshape(batch_size, channel // self.arg_in_dim, -1, self.arg_in_dim)

        out = self.inner_net(input)

        out = out.reshape(batch_size, -1, input_size, input_size)
        return out

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self._inner_net_forward(out)
        out = self.bn2(self.conv2(out))
        out = self._inner_net_forward(out)
        out += self.shortcut(x)
        return out


class ResNet_Xor(nn.Module):
    def __init__(self, block, config, num_classes=10):
        super(ResNet_Xor, self).__init__()
        self.in_planes = 16
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_channel
        self.num_blocks = config.model.num_blocks

        self.loss_func = nn.CrossEntropyLoss()

        if config.model.inner_net == 'quad':
            self.inner_net = QuadraticInnerNet()
            self.arg_in_dim = 2

        else:
            self.inner_net = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(self.in_hidden_dim, 1))]))

        self.conv1 = nn.Conv2d(3, self.in_planes * self.arg_in_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes * self.arg_in_dim)

        self.layer1 = self._make_layer(block, 16, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, self.num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inner_net, self.in_planes, planes, self.arg_in_dim, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _inner_net_forward(self, x):
        batch_size = x.shape[0]
        channel = x.shape[1]
        input_size = x.shape[-1]

        input = x.reshape(batch_size, channel // self.arg_in_dim, -1, self.arg_in_dim)

        out = self.inner_net(input)

        out = out.reshape(batch_size, -1, input_size, input_size)
        return out

    def forward(self, x, labels):
        out = self._inner_net_forward(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1)

        out = self.linear(out)

        loss = self.loss_func(out, labels)

        return out, loss, None


def ResNet20_Xor(config):
    return ResNet_Xor(BasicBlock_InnerNet, config)
