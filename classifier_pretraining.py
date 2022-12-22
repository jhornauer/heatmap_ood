import os
import random
import pickle
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from optimization import train_classifier
from evaluation import test_classifier
from selection import select_transform, select_dataset, select_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, default='cifar10',
                        help='cifar10, cifar100 or tinyimagenet')
    parser.add_argument('-arch', type=str, required=True, default='resnet18',
                        help='resnet18, resnet50 or wideresnet')
    parser.add_argument('-device', type=str, default='cuda:0', help='device')
    parser.add_argument('-valid_split', type=float)
    parser.add_argument('-depth', type=int, default=40)
    parser.add_argument('-width', type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
    else:
        num_classes = 10

    lr = 0.1
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        n_epochs = 100
        batch_size = 200
    elif args.dataset == 'tinyimagenet':
        n_epochs = 200
        batch_size = 128
    else:
        raise NotImplementedError

    train_transform = select_transform(args.dataset, pretrain=True)
    test_transform = select_transform(args.dataset, pretrain=False)
    dataset = select_dataset(args.dataset, train_transform, train=True)
    testset = select_dataset(args.dataset, test_transform, train=False)

    # setup validation split
    len_dataset = len(dataset)
    model_name = str(args.dataset) + '_' + str(args.arch)
    valid_factor = 0.01

    dataset_idx = list(range(len_dataset))
    valid_idx = sorted(random.sample(dataset_idx, k=int(len(dataset) * valid_factor)))
    train_idx = [x for x in dataset_idx if x not in valid_idx]
    trainset = torch.utils.data.Subset(dataset, train_idx)
    validset = torch.utils.data.Subset(dataset, valid_idx)

    if args.arch == 'wideresnet':
        if args.depth != 28 or args.width != 10:
            model_name = model_name + '_' + str(args.depth) + '_' + str(args.width)

    # save train and valid idx
    pickle.dump((train_idx, valid_idx), open(os.getcwd() + '/Models/' + model_name + '.p', 'wb'))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    net, cls = select_classifier(args.dataset, args.arch, num_classes, depth=args.depth, width=args.width)
    net.to(device)
    cls.to(device)

    optimizer = torch.optim.SGD(list(net.parameters()) + list(cls.parameters()), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    train_accs, valid_accs = train_classifier(net, cls, trainloader, validloader, optimizer, scheduler, n_epochs,
                                              device, model_name)
    test_acc = test_classifier(net, cls, testloader, device)
    print('Test Accuracy: ', test_acc)

    plt.plot(train_accs)
    plt.plot(valid_accs)
    plt.plot(n_epochs, test_acc, 'x')
    plt.legend(['Train-split', 'Valid-split', 'Test-split'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
