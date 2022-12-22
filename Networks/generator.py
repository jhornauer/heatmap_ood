import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18_CIFAR10_Deconv_GAN(nn.Module):
    def __init__(self, latent_dim=100, fw_layers=1, num_classes=10):
        super(ResNet18_CIFAR10_Deconv_GAN, self).__init__()
        self.latent_dim = latent_dim
        self.fw_layers = fw_layers
        self.num_classes = num_classes
        self.num_filters = [512, 256, 128, 64]

        self.fc1 = nn.Linear(latent_dim, 100)

        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(100, self.num_filters[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU(True))
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[0], self.num_filters[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[1]),
            nn.ReLU(True))
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[1], self.num_filters[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[2]),
            nn.ReLU(True))
        self.conv_block4 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[2], self.num_filters[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[3]),
            nn.ReLU(True))
        self.conv_block5 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[3], 3, 1, 1, 0, bias=False),
            nn.Tanh())

    def forward(self, z, y_hat):
        if self.fw_layers > 1:
            x = z[0]
        else:
            x = z
        ys = F.one_hot(y_hat, num_classes=self.num_classes)
        x = torch.cat((x, ys), dim=1)
        x = self.fc1(x)
        x = x.view(-1, 100, 1, 1)
        x = self.conv_block1(x)
        if self.fw_layers >= 2:
            x = x + z[1]
        x = self.conv_block2(x)
        if self.fw_layers >= 3:
            x = x + z[2]
        x = self.conv_block3(x)
        if self.fw_layers == 4:
            x = x + z[3]
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class ResNet18_CIFAR100_Deconv_GAN(nn.Module):
    def __init__(self, latent_dim=512, fw_layers=1, num_classes=100):
        super(ResNet18_CIFAR100_Deconv_GAN, self).__init__()
        self.latent_dim = latent_dim
        self.fw_layers = fw_layers
        self.num_classes = num_classes
        self.num_filters = [512, 256, 128, 64]

        self.fc1 = nn.Linear(latent_dim, 512)
        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, self.num_filters[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU())
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[0], self.num_filters[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[1]),
            nn.ReLU())
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[1], self.num_filters[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[2]),
            nn.ReLU())
        self.conv_block4 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[2], self.num_filters[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[3]),
            nn.ReLU())
        self.conv_block5 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[3], 3, 1, 1, 0, bias=False),
            nn.Tanh())

    def forward(self, z, y_hat):
        if self.fw_layers > 1:
            x = z[0]
        else:
            x = z
        y_hat = F.one_hot(y_hat, num_classes=self.num_classes)
        x = torch.cat((x, y_hat), dim=1)
        x = self.fc1(x)
        x = x.view(-1, 512, 1, 1)
        x = self.conv_block1(x)
        if self.fw_layers >= 2:
            x = x + z[1]
        x = self.conv_block2(x)
        if self.fw_layers >= 3:
            x = x + z[2]
        x = self.conv_block3(x)
        if self.fw_layers == 4:
            x = x + z[3]
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x


class TImageNet_Deconv_GAN(nn.Module):
    def __init__(self, latent_dim=512, fw_layers=1, num_classes=200):
        super(TImageNet_Deconv_GAN, self).__init__()
        self.latent_dim = latent_dim
        self.fw_layers = fw_layers
        self.num_classes = num_classes
        self.num_filters = [2048, 1024, 512, 256, 128]

        self.fc1 = nn.Linear(latent_dim, 512)
        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, self.num_filters[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU(True))
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[0], self.num_filters[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[1]),
            nn.ReLU(True))
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[1], self.num_filters[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[2]),
            nn.ReLU(True))
        self.conv_block4 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[2], self.num_filters[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[3]),
            nn.ReLU(True))
        self.conv_block5 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[3], self.num_filters[4], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[4]),
            nn.ReLU(True))
        self.conv_block6 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[4], 3, 1, 1, 0, bias=False),
            nn.Tanh())

    def forward(self, z, y_hat):
        if self.fw_layers > 1:
            x = z[0]
        else:
            x = z
        y_hat = F.one_hot(y_hat, num_classes=self.num_classes)
        x = torch.cat((x, y_hat), dim=1)
        x = self.fc1(x)
        x = x.view(-1, 512, 1, 1)
        x = self.conv_block1(x)
        if self.fw_layers >= 2:
            x = x + z[1]
        x = self.conv_block2(x)
        if self.fw_layers >= 3:
            x = x + z[2]
        x = self.conv_block3(x)
        if self.fw_layers == 4:
            x = x + z[3]
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        return x


class WideResNet_40x2_Deconv_GAN(nn.Module):
    def __init__(self, latent_dim=128, fw_layers=1, num_classes=10):
        super(WideResNet_40x2_Deconv_GAN, self).__init__()
        self.latent_dim = latent_dim
        self.fw_layers = fw_layers
        self.num_classes = num_classes
        self.num_filters = [256, 128, 64, 32]

        self.fc1 = nn.Linear(latent_dim, 128)
        self.conv_block1 = nn.Sequential(
            nn.ConvTranspose2d(128, self.num_filters[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.num_filters[0]),
            nn.ReLU(True))
        self.conv_block2 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[0], self.num_filters[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[1]),
            nn.ReLU(True))
        self.conv_block3 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[1], self.num_filters[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[2]),
            nn.ReLU(True))
        self.conv_block4 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[2], self.num_filters[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.num_filters[3]),
            nn.ReLU(True))
        self.conv_block5 = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters[3], 3, 1, 1, 0, bias=False),
            nn.Tanh())

    def forward(self, z, y_hat):
        if self.fw_layers > 1:
            x = z[0]
        else:
            x = z
        ys = F.one_hot(y_hat, num_classes=self.num_classes)
        x = torch.cat((x, ys), dim=1)
        x = self.fc1(x)
        x = x.view(-1, 128, 1, 1)
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        if self.fw_layers >= 2:
            x2 = x2 + z[1]
        x3 = self.conv_block3(x2)
        if self.fw_layers >= 3:
            x3 = x3 + z[2]
        x4 = self.conv_block4(x3)
        if self.fw_layers == 4:
            x4 = x4 + z[3]
        x5 = self.conv_block5(x4)
        return x5
