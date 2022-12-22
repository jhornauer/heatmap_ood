import os
import argparse
import torch
from torch.utils.data import DataLoader
from evaluation import test_classifier
from selection import select_transform, select_dataset, select_classifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True, default='cifar10',
                        help='cifar10 cifar100 or tinyimagenet')
    parser.add_argument('-arch', type=str, required=True, default='resnet18',
                        help='resnet18, resnet50 or wideresnet')
    parser.add_argument('-device', type=str, default='cuda:0', help='device')
    parser.add_argument('-depth', type=int, default=28)
    parser.add_argument('-width', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    batch_size = 200

    if args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tinyimagenet':
        num_classes = 200
    else:
        num_classes = 10

    test_transform = select_transform(args.dataset, pretrain=False)
    testset = select_dataset(args.dataset, test_transform, train=False)

    model_name = str(args.dataset) + '_' + str(args.arch)

    if args.arch == 'wideresnet':
        model_name = model_name + '_' + str(args.depth) + '_' + str(args.width)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    net, cls = select_classifier(args.dataset, args.arch, num_classes, depth=args.depth, width=args.width)
    net.to(device)
    cls.to(device)
    checkpoint = torch.load(os.getcwd() + '/Models/' + model_name + '.pt', map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    cls.load_state_dict(checkpoint['cls_state_dict'])

    test_acc = test_classifier(net, cls, testloader, device)
    print('Test Accuracy: ', test_acc)
    