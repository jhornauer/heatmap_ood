import os
from torchvision import datasets, transforms
from Networks.resnet import ResNet18, ResNet50, ResNet_Linear
from Networks.wideresnet import Wide_ResNet, WideResNet_Linear
from Networks.generator import ResNet18_CIFAR10_Deconv_GAN, ResNet18_CIFAR100_Deconv_GAN, TImageNet_Deconv_GAN, \
    WideResNet_40x2_Deconv_GAN


def select_transform(dataset, pretrain=False):
    if pretrain:
        if dataset in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset == 'tinyimagenet':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise NotImplementedError
    else:
        if dataset in ['cifar10', 'cifar100', 'tinyimagenet']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise NotImplementedError

    return transform


def select_dataset(dataset, transform, train):
    if dataset == 'cifar10':
        trainset = datasets.CIFAR10(os.getcwd() + '/Datasets/', train=train, transform=transform)
    elif dataset == 'cifar100':
        trainset = datasets.CIFAR100(os.getcwd() + '/Datasets/', train=train, transform=transform)
    elif dataset == 'tinyimagenet':
        if train:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/tiny-imagenet-200/train/',
                                            transform=transform)
        else:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/tiny-imagenet-200/test/',
                                            transform=transform)
    elif dataset == 'tinyimages':
        if train:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/TinyImages-100000/', transform=transform)
        else:
            raise RuntimeError('Only available as ood training set')
    elif dataset == 'places365':
        if train:
            trainset = datasets.ImageFolder(os.getcwd() + '/Datasets/Places365/', transform=transform)
        else:
            raise RuntimeError('Only available as ood training set')
    else:
        raise NotImplementedError
    return trainset


def select_classifier(dataset, arch, num_classes, fw_layers=1, depth=40, width=2):
    if dataset == 'cifar10':
        if arch == 'resnet18':
            enc = ResNet18(fw_layers=fw_layers)
            cls = ResNet_Linear()
        elif arch == 'wideresnet':
            enc = Wide_ResNet(depth, width, 0.3, fw_layers=fw_layers)
            cls = WideResNet_Linear(width, num_classes=num_classes)
        else:
            raise NotImplementedError
    elif dataset == 'cifar100':
        if arch == 'resnet18':
            enc = ResNet18(fw_layers=fw_layers)
            cls = ResNet_Linear(num_classes=num_classes)
        elif arch == 'wideresnet':
            enc = Wide_ResNet(depth, width, 0.3, fw_layers=fw_layers)
            cls = WideResNet_Linear(width, num_classes=num_classes)
        else:
            raise NotImplementedError
    elif dataset == 'tinyimagenet':
        if arch == 'resnet50':
            enc = ResNet50(fw_layers=fw_layers, img_size=64)
            cls = ResNet_Linear(num_classes=num_classes, expansion=4)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return enc, cls


def select_generator(dataset, arch, z_dim, fw_layers, num_classes, depth=None, width=None):
    if dataset == 'cifar10':
        if arch == 'resnet18':
            gen = ResNet18_CIFAR10_Deconv_GAN(latent_dim=z_dim, fw_layers=fw_layers, num_classes=num_classes)
        elif arch == 'wideresnet':
            if depth == 40 and width == 2:
                gen = WideResNet_40x2_Deconv_GAN(latent_dim=z_dim, fw_layers=fw_layers, num_classes=num_classes)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif dataset == 'cifar100':
        if arch == 'resnet18':
            gen = ResNet18_CIFAR100_Deconv_GAN(latent_dim=z_dim, fw_layers=fw_layers, num_classes=num_classes)
        elif arch == 'wideresnet':
            if depth == 40 and width == 2:
                gen = WideResNet_40x2_Deconv_GAN(latent_dim=z_dim, fw_layers=fw_layers, num_classes=num_classes)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif dataset == 'tinyimagenet':
        if arch == 'resnet50':
            gen = TImageNet_Deconv_GAN(latent_dim=z_dim, fw_layers=fw_layers, num_classes=num_classes)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return gen


def select_ood_testset(dataset, transform):
    if dataset == 'textures':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/Textures/dtd/images/', transform=transform)
    elif dataset == 'svhn':
        ood_data = datasets.SVHN(os.getcwd() + '/Datasets/SVHN/', split='test', transform=transform)
    elif dataset == 'places365':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/Places365/', transform=transform)
    elif dataset == 'lsun-c':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/LSUN/', transform=transform)
    elif dataset == 'lsun-r':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/LSUN_resize/', transform=transform)
    elif dataset == 'isun':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/iSUN/', transform=transform)
    elif dataset == 'cifar10':
        ood_data = datasets.CIFAR10(os.getcwd() + '/Datasets/', transform=transform, train=False)
    elif dataset == 'cifar100':
        ood_data = datasets.CIFAR100(os.getcwd() + '/Datasets/', transform=transform, train=False)
    elif dataset == 'tinyimagenet':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/tiny-imagenet-200/test/', transform=transform)
    elif dataset == 'lsun':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/SUN/', transform=transform)
    elif dataset == 'inaturalist':
        ood_data = datasets.ImageFolder(os.getcwd() + '/Datasets/iNaturalist/', transform=transform)
    else:
        raise NotImplementedError
    return ood_data


def select_ood_transform(dataset, id_dataset='cifar10'):
    if id_dataset in ['cifar10', 'cifar100']:
        if dataset in ['textures', 'places365', 'lsun-c']:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif dataset in ['svhn', 'lsun-r', 'isun', 'cifar10', 'cifar100', 'tinyimages']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            raise NotImplementedError
    elif id_dataset == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise NotImplementedError
    return transform
