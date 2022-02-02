import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from collections import OrderedDict

from torch.autograd import Variable

from model.xorneuron import QuadraticInnerNet
from model.conv2dlayer import Conv2dLayerWithComplexNeuronsForResnet, Conv2dLayerWithComplexNeurons, \
    Conv2dLayerWithQuadraticFuction

__all__ = ['Xor_ResNet', 'Xorresnet20', 'Xor_resnet_basicblock', 'ResNet20_Xor', 'BasicBlock_InnerNet']


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


class Xor_resnet_basicblock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, inner_net, stride=1, option='A'):
        super(Xor_resnet_basicblock, self).__init__()

        self.conv1 = Conv2dLayerWithQuadraticFuction(inner_net,
                                                     arity=2,
                                                     in_channels=in_planes,
                                                     out_channels=planes,
                                                     kernel_size=3,
                                                     stride=stride,
                                                     padding=1,
                                                     bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = Conv2dLayerWithQuadraticFuction(inner_net,
                                                     arity=2,
                                                     in_channels=planes,
                                                     out_channels=planes,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != planes:
            self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        skip_input = self.shortcut(x)
        out, _ = self.conv1(x)
        out += skip_input
        out, _ = self.conv2(out)

        return out


class Xor_ResNet(nn.Module):
    def __init__(self, block, config):
        super(Xor_ResNet, self).__init__()

        self.num_cell_types = config.model.num_cell_types
        self.input_channel = config.model.input_channel
        self.arg_in_dim = config.model.arg_in_dim
        self.in_channel = config.model.in_channel
        self.num_classes = config.model.num_classes
        self.num_blocks = config.model.num_blocks

        self.in_planes = 16

        self.loss_func = nn.CrossEntropyLoss()

        self.inner_net = QuadraticInnerNet()

        self.conv1 = Conv2dLayerWithQuadraticFuction(self.inner_net,
                                                     arity=2,
                                                     in_channels=self.input_channel,
                                                     out_channels=self.in_planes,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     bias=False)

        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, self.num_blocks[2], stride=2)
        self.linear = nn.Linear(64, self.num_classes)

        self._init_param()

    def _init_param(self):
        mlp_modules = [
            xx for xx in [self.linear, self.inner_net] if xx is not None
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
                                nn.init.xavier_uniform_(mmm.weight.data)
                                if mmm.bias is not None:
                                    mmm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.inner_net, stride, option='A'))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, labels):
        # print(f'x_1: {x.shape}')
        out, _ = self.conv1(x)
        # print(f'x_2: {out.shape}')

        # print(f'block1 input: {out.shape}')
        out = self.layer1(out)
        # print(f'block1 output & block2 input: {out.shape}')
        out = self.layer2(out)
        # print(f'block2 output & block3 input: {out.shape}')
        out = self.layer3(out)
        # print(f'block3 output: {out.shape}')

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        loss = self.loss_func(out, labels)

        return out, loss, None


class BasicBlock_InnerNet(nn.Module):
    expansion = 1

    def __init__(self, inner_net, in_planes, planes, stride=1, option="A"):
        super(BasicBlock_InnerNet, self).__init__()
        self.inner_net = inner_net

        self.conv1 = nn.Conv2d(
            in_planes, planes * 2, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(
            planes, planes * 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def _inner_net_forward(self, x):
        batch_size = x.shape[0]
        channel = x.shape[1]
        input_size = x.shape[-1]

        input = x.reshape(batch_size, channel // 2, -1, 2)

        out = self.inner_net(input)

        out = out.reshape(batch_size, -1, input_size, input_size)
        return out

    def forward(self, x):
        out = self._inner_net_forward(self.bn1(self.conv1(x)))
        out = self._inner_net_forward(self.bn2(self.conv2(out)))
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

        self.inner_net = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.arg_in_dim, self.in_hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.in_hidden_dim, self.in_hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(self.in_hidden_dim, 1))]))

        self.conv1 = nn.Conv2d(3, self.in_planes * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes * 2)

        self.layer1 = self._make_layer(block, 16, self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, self.num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inner_net, self.in_planes, planes, stride))
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


def Xorresnet20(config):
    return Xor_ResNet(Xor_resnet_basicblock, config)


def ResNet20_Xor(config):
    return ResNet_Xor(BasicBlock_InnerNet, config)
