import os
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import set_requires_grad
from evaluation import test_classifier
from classifier_utils import get_classifier_features


def ase_loss_weighted(output, input, device):
    diff = (output - input) ** 2
    weights = torch.ones(diff.size(), device=device)
    weights = weights + torch.abs(input) * 5
    anomaly_score = torch.sum(weights * diff, dim=(tuple(range(1, output.dim()))))
    return anomaly_score


def train_classifier(net, cls, trainloader, validloader, optimizer, scheduler, n_epochs, device, model_name):
    train_accs = []
    valid_accs = []
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    net.train()
    cls.train()
    for e in range(n_epochs):
        net.train()
        cls.train()
        running_acc = 0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = cls(net(images))
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_acc += (torch.argmax(outputs, dim=1) == labels).float().sum().item()

        running_acc = running_acc / len(trainloader.dataset)
        train_accs.append(running_acc)

        running_acc_valid = test_classifier(net, cls, validloader, device)
        valid_accs.append(running_acc_valid)
        if best_acc < running_acc_valid:
            best_acc = running_acc_valid
            mapping = {'net_state_dict': copy.deepcopy(net.state_dict()),
                       'cls_state_dict': copy.deepcopy(cls.state_dict())}
            torch.save(mapping, os.getcwd() + '/Models/' + model_name + '.pt')

        scheduler.step(running_acc_valid)
        print('Epoch: ' + str(e) + ' -- Accuracy: ' + str(running_acc_valid))
    return train_accs, valid_accs


def learn_prototype(encoder, classifier, generator, optimizer_g, n_epochs, trainset, batch_size, min_val, max_val,
                    device, fw_layers=1, save_dir=None):
    encoder.eval()
    classifier.eval()
    generator.train()

    gan_dict = {"min": min_val, "max": max_val}

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    for e in range(n_epochs):
        running_loss_gen = 0
        # set requires grad False
        set_requires_grad(encoder, False)
        set_requires_grad(classifier, False)
        for i, (images, gt_images) in enumerate(trainloader):
            images = images.to(device)
            gt_images = gt_images.to(device)

            # train generator
            optimizer_g.zero_grad()

            zs_real, ys_real = get_classifier_features(encoder, classifier, images, fw_layers)

            for l_idx in range(fw_layers):
                zs_real[l_idx] = (zs_real[l_idx] - min_val) / (max_val - min_val)

            x_hat = generator(zs_real, ys_real)

            x_hat = x_hat + images
            g_loss = torch.mean(ase_loss_weighted(x_hat, gt_images, device))

            running_loss_gen += g_loss.item()

            g_loss.backward()
            optimizer_g.step()

        running_loss_gen = running_loss_gen / len(trainset)

        print('Epoch {} -- Reconstruction Loss: {:.2f}'.format(e, running_loss_gen))

        if (e + 1) % 10 == 0:
            mapping = {"generator": copy.deepcopy(generator.state_dict())}
            gan_dict[e] = mapping

    torch.save(gan_dict, save_dir)
    return generator