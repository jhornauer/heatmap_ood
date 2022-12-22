import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, fw_layers=1, img_size=32):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.fw_layers = fw_layers
        self.img_size = img_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        if self.img_size == 64:
            self.maxpool = nn.MaxPool2d(kernel_size=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        if self.img_size == 64:
            x1 = self.maxpool(x1)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        xr = self.avgpool(x5)
        xr = xr.view(xr.size(0), -1)
        if self.fw_layers == 4:
            return xr, x5, x4, x3
        elif self.fw_layers == 3:
            return xr, x5, x4
        elif self.fw_layers == 2:
            return xr, x5
        else:
            return xr

    def forward_threshold(self, x, threshold=1e10):
        x1 = F.relu(self.bn1(self.conv1(x)))
        if self.img_size == 64:
            x1 = self.maxpool(x1)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        xr = self.avgpool(x5)
        xr = xr.clip(max=threshold)
        xr = xr.view(xr.size(0), -1)
        return xr

    def feature_list(self, x):
        out_list = []
        x = F.relu(self.bn1(self.conv1(x)))
        out_list.append(x)
        x = self.layer1(x)
        out_list.append(x)
        x = self.layer2(x)
        out_list.append(x)
        x = self.layer3(x)
        out_list.append(x)
        x = self.layer4(x)
        out_list.append(x)
        return out_list

    def intermediate_forward(self, x, layer_index):
        x = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            x = self.layer1(x)
        elif layer_index == 2:
            x = self.layer1(x)
            x = self.layer2(x)
        elif layer_index == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif layer_index == 4:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        return x


def ResNet18(fw_layers=1, img_size=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], fw_layers=fw_layers, img_size=img_size)


def ResNet50(fw_layers=1, img_size=32):
    return ResNet(Bottleneck, [3, 4, 6, 3], fw_layers=fw_layers, img_size=img_size)


class ResNet_Linear(nn.Module):
    def __init__(self, num_classes=10, expansion=1):
        super(ResNet_Linear, self).__init__()

        self.expansion = expansion
        self.linear = nn.Linear(512 * self.expansion, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
