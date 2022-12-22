# Source: https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, fw_layers=1):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.fw_layers = fw_layers

        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.n_channels = nStages[3]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.avgpool = nn.AvgPool2d(kernel_size=8)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        xr = F.relu(self.bn1(x4))
        xr = self.avgpool(xr)
        xr = xr.view(xr.size(0), -1)
        if self.fw_layers == 4:
            return xr, x4, x3, x2
        elif self.fw_layers == 3:
            return xr, x4, x3
        elif self.fw_layers == 2:
            return xr, x4
        else:
            return xr

    def forward_threshold(self, x, threshold=1e10):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        xr = F.relu(self.bn1(x4))
        xr = self.avgpool(xr)
        xr = xr.clip(max=threshold)
        xr = xr.view(xr.size(0), -1)
        return xr

    def intermediate_forward(self, x, layer_index):
        x = self.conv1(x)
        if layer_index == 1:
            x = self.layer1(x)
        elif layer_index == 2:
            x = self.layer1(x)
            x = self.layer2(x)
        elif layer_index == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = F.relu(self.bn1(x))
        return x

    def feature_list(self, x):
        out_list = []
        x = self.conv1(x)
        out_list.append(x)
        x = self.layer1(x)
        out_list.append(x)
        x = self.layer2(x)
        out_list.append(x)
        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        out_list.append(x)
        # out_list.append(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.n_channels)
        return out_list


class WideResNet_Linear(nn.Module):
    def __init__(self, widen_factor, num_classes=10, img_res=32):
        super(WideResNet_Linear, self).__init__()

        if img_res == 32:
            self.linear = nn.Linear(64*widen_factor, num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.linear(x)
        return out
