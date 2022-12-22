import os
import json
import argparse
import torch
from torch import optim
from torchvision import transforms
from utils import merge_dicts, make_checkpoint, dict2clsattr
from selection import select_dataset, select_classifier, select_generator
from classifier_utils import get_norm_values
from prototype_heatmap import Prototype_Heatmap
from optimization import learn_prototype


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./configs/cifar10_resnet18_ood.json')
    parser.add_argument('-device', type=str, default='cuda:0', help='device')
    parser.add_argument('-fw_layers', type=int, default=1, help='number of input layers to generator')
    parser.add_argument('-num_ood', type=int)

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as f:
            model_config = json.load(f)
        train_config = vars(args)
    else:
        raise NotImplementedError

    config_dict = merge_dicts(model_config, train_config)
    config_dict["batch_size"] = 200
    config = dict2clsattr(config_dict)

    config.z_dim = config.z_dim + config.num_classes
    config_dict["z_dim"] = config.z_dim

    device = torch.device(config.device)

    load_dir = str(config.id_dataset) + '_' + str(config.arch)
    if config.arch == 'wideresnet':
        if config.depth != 28 or config.width != 10:
            load_dir = load_dir + '_' + str(config.depth) + '_' + str(config.width)
    checkpoint_dir = make_checkpoint(config_dict, setup='proto')
    model_dir = checkpoint_dir + '/model.pt'
    print('Saving to: ', checkpoint_dir)

    id_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if config.id_dataset in ['cifar10', 'cifar100']:
        ood_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif config.id_dataset == 'tinyimagenet':
        ood_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise NotImplementedError

    id_dataset = select_dataset(config.id_dataset, id_transform, train=True)
    ood_dataset = select_dataset(config.ood_dataset, ood_transform, train=True)

    checkpoint = torch.load(os.getcwd() + '/Models/' + load_dir + '.pt', map_location=device)

    enc, cls = select_classifier(config.id_dataset, config.arch, config.num_classes, config.fw_layers, config.depth,
                                 config.width)
    enc.to(device)
    cls.to(device)
    enc.load_state_dict(checkpoint['net_state_dict'])
    cls.load_state_dict(checkpoint['cls_state_dict'])
    norm_min, norm_max = get_norm_values(enc, cls, id_dataset, config.batch_size, device, fw_layers=config.fw_layers)

    gen = select_generator(config.id_dataset, config.arch, config.z_dim, config.fw_layers, config.num_classes,
                           depth=config.depth, width=config.width)
    gen.to(device)
    gen_optimizer = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.b1, config.b2))

    dataset = Prototype_Heatmap(id_dataset, ood_dataset, enc, cls, device, config.num_classes, config.num_ood,
                                fw_layers=config.fw_layers)

    learn_prototype(enc, cls, gen, gen_optimizer, config.n_epochs, dataset, config.batch_size, norm_min, norm_max,
                    device, fw_layers=config.fw_layers, save_dir=model_dir)
