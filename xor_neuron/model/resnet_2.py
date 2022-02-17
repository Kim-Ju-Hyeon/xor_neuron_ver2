import torch
import torch.nn as nn
import os

from model.xorneuron import QuadraticInnerNet

from collections import OrderedDict

__all__ = [
    "ResNet_Xor_2",
    "resnet18",
    "resnet34",
    "resnet50",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inner_net,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        arg_in_dim=2
    ):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.arg_in_dim = arg_in_dim
        self.inner_net = inner_net
        self.inner_net_conv = conv1x1(planes, planes*self.arg_in_dim)
        self.inner_net_bn = norm_layer(planes*self.arg_in_dim)

        self.conv1 = conv3x3(inplanes, planes*self.arg_in_dim, stride)
        self.bn1 = norm_layer(planes*self.arg_in_dim)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def _inner_net_forward(self, x):
        batch_size = x.shape[0]
        channel = x.shape[1]
        input_size = x.shape[-1]

        input = x.reshape(batch_size, channel // self.arg_in_dim, -1, self.arg_in_dim)
        out = self.inner_net(input)

        out = out.reshape(batch_size, -1, input_size, input_size)
        return out

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._inner_net_forward(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.inner_net_bn(self.inner_net_conv(out))
        out = self._inner_net_forward(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inner_net,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        arg_in_dim=2
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.arg_in_dim = arg_in_dim
        self.inner_net = inner_net
        self.inner_net_conv = conv1x1(planes*self.expansion, planes*self.expansion*self.arg_in_dim)
        self.inner_net_bn = norm_layer(planes*self.expansion*self.arg_in_dim)

        self.conv1 = conv1x1(inplanes, width*self.arg_in_dim)
        self.bn1 = norm_layer(width*self.arg_in_dim)

        self.conv2 = conv3x3(width, width*self.arg_in_dim, stride, groups, dilation)
        self.bn2 = norm_layer(width*self.arg_in_dim)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def _inner_net_forward(self, x):
        batch_size = x.shape[0]
        channel = x.shape[1]
        input_size = x.shape[-1]

        input = x.reshape(batch_size, channel // self.arg_in_dim, -1, self.arg_in_dim)
        out = self.inner_net(input)

        out = out.reshape(batch_size, -1, input_size, input_size)
        return out

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._inner_net_forward(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._inner_net_forward(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.inner_net_bn(self.inner_net_conv(out))
        out = self._inner_net_forward(out)

        return out


class ResNet_Xor_2(nn.Module):
    def __init__(
        self,
        block,
        layers,
        config,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet_Xor_2, self).__init__()
        self.config = config
        self.arg_in_dim = config.model.arg_in_dim
        self.in_hidden_dim = config.model.in_channel

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

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes*self.arg_in_dim, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes*self.arg_in_dim)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self._inner_net_forward(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        loss = self.loss_func(x, labels)

        return x, loss, None


def resnet18(config):
    return ResNet_Xor_2(BasicBlock, [2, 2, 2, 2], config)


def resnet34(config):
    return ResNet_Xor_2(BasicBlock, [3, 4, 6, 3], config)


def resnet50(config):
    return ResNet_Xor_2(Bottleneck, [3, 4, 6, 3], config)
