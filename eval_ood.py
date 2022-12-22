import os
import argparse
import json
import random
import torch
import numpy as np
from evaluation import test_generator, get_metrics_ood, results2file_ood
from utils import merge_dicts, dict2clsattr, search_checkpoint
from selection import select_classifier, select_generator, select_ood_transform, select_ood_testset
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='./configs/cifar10_resnet18_ood.json')
    parser.add_argument('-device', type=str, default='cuda:0', help='device')
    parser.add_argument('-fw_layers', type=int, default=1, help='number of input layers to generator')
    parser.add_argument('-checkpoint_name', type=str)

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
    if args.checkpoint_name is None:
        checkpoint_dir = search_checkpoint(config_dict)
    else:
        checkpoint_dir = os.getcwd() + '/checkpoints/' + config.checkpoint_name
    result_dir = checkpoint_dir + '/results.txt'
    model_dir = checkpoint_dir + '/model.pt'

    id_transform = select_ood_transform(config.id_dataset, config.id_dataset)
    id_dataset = select_ood_testset(config.id_dataset, id_transform)

    num_ood = len(id_dataset) // 5

    if config.id_dataset in ['cifar10', 'cifar100']:
        ood_testsets = ['isun', 'textures', 'svhn', 'places365', 'lsun-c', 'lsun-r']
        n_testsets = 6
    elif config.id_dataset == 'tinyimagenet':
        ood_testsets = ['textures', 'inaturalist', 'lsun']
        n_testsets = 3
    else:
        raise NotImplementedError

    checkpoint = torch.load(os.getcwd() + '/Models/' + load_dir + '.pt')
    checkpoint_gan = torch.load(model_dir)

    enc, cls = select_classifier(config.id_dataset, config.arch, config.num_classes, config.fw_layers,
                                 depth=config.depth, width=config.width)
    enc.to(device)
    cls.to(device)
    enc.load_state_dict(checkpoint['net_state_dict'])
    cls.load_state_dict(checkpoint['cls_state_dict'])

    gen = select_generator(config.id_dataset, config.arch, config.z_dim, config.fw_layers, config.num_classes,
                           depth=config.depth, width=config.width)
    gen.to(device)
    gen.load_state_dict(checkpoint_gan[149]['generator'])

    norm_min = checkpoint_gan['min']
    norm_max = checkpoint_gan['max']

    scores_id = test_generator(enc, cls, gen, id_dataset, config.batch_size, norm_min, norm_max, device,
                               fw_layers=config.fw_layers)

    avg_result_dict = {"rocauc": 0, "aupr_success": 0, "aupr_error": 0, "fpr": 0}

    for ood_set in ood_testsets:
        ood_transform = select_ood_transform(ood_set, config.id_dataset)
        ood_dataset = select_ood_testset(ood_set, ood_transform)
        scores_ood = test_generator(enc, cls, gen, ood_dataset, config.batch_size, norm_min, norm_max, device,
                                    fw_layers=config.fw_layers, n_samples=num_ood)

        scores = np.concatenate([scores_id, scores_ood[:num_ood]], axis=0)
        labels = np.concatenate([np.ones(len(id_dataset)), np.zeros(num_ood)], axis=0)

        results_dict = get_metrics_ood(labels, scores, invert_score=True)
        results2file_ood(result_dir, results_dict, ood_set)

        avg_result_dict["rocauc"] += results_dict["rocauc"]
        avg_result_dict["aupr_success"] += results_dict["aupr_success"]
        avg_result_dict["aupr_error"] += results_dict["aupr_error"]
        avg_result_dict["fpr"] += results_dict["fpr"]

    avg_result_dict["rocauc"] = avg_result_dict["rocauc"] / n_testsets
    avg_result_dict["aupr_success"] = avg_result_dict["aupr_success"] / n_testsets
    avg_result_dict["aupr_error"] = avg_result_dict["aupr_error"] / n_testsets
    avg_result_dict["fpr"] = avg_result_dict["fpr"] / n_testsets
    results2file_ood(result_dir, avg_result_dict, 'average')
