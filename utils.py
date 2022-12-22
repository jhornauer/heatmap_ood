import os
from datetime import datetime


#######################################################################
def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(config):
    cfgs = {}
    for k, v in config.items():
        cfgs[k] = v

    class cfg_container: pass

    cfg_container.config = config
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container

#######################################################################


def merge_dicts(dict_a, dict_b):
    dict_a = flatten_dict(dict_a)
    dict_a.update({k: v for k, v in dict_b.items() if v is not None})
    return dict_a


def make_checkpoint(config, setup='proto'):
    config['setup'] = setup
    root_dir = os.getcwd() + '/checkpoints/'
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    checkpoint_dir = root_dir + config["id_dataset"] + '_' + config["arch"]
    checkpoint_dir = checkpoint_dir + '_ood'

    checkpoint_dir = checkpoint_dir + '_fw' + str(config["fw_layers"]) + '_' + timestamp

    os.mkdir(checkpoint_dir)
    with open(checkpoint_dir + '/config.txt', 'w') as f:
        print(config, file=f)
    return checkpoint_dir


def search_checkpoint(config):
    root_dir = os.getcwd() + '/checkpoints/'
    checkpoints = os.listdir(root_dir)
    checkpoints_filtered = [f for f in checkpoints if config["id_dataset"] in f and config["arch"] in f]
    checkpoints_latest = sorted(checkpoints_filtered, reverse=True)[0]
    return root_dir + checkpoints_latest


def set_requires_grad(model, value):
    for name, param in model.named_parameters():
        param.requires_grad = value
