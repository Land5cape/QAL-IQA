#coding=utf-8
import os

import torch
import torch.nn as nn
from . import vgg, resnet

weights_dir = os.path.dirname(os.path.dirname(__file__)) + '\weights'


def BuildAutoEncoder(arch, load_weights=True, mid_f=False, device=torch.device('cpu')):
    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(arch)
        model = vgg.VGGAutoEncoder(configs, mid_f=mid_f)

    elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck, mid_f=mid_f)

    else:
        return None

    model = nn.DataParallel(model).to(device)

    if load_weights:

        weights = {
            'vgg16': weights_dir + '/imagenet-vgg16.pth',
            # 'vgg16' : weights_dir + 'weights/objects365-vgg16.pth',
            'vgg19': weights_dir + '/objects365-vgg16.pth',
            'resnet50': weights_dir + '/objects365-resnet50.pth',
            'resnet101': weights_dir + '/objects365-resnet101.pth',
        }

        if arch in weights.keys():
            print('load_weights')
            weights_path = weights[arch]
            checkpoint = torch.load(weights_path)
            model_dict = model.state_dict()
            model_dict.update(checkpoint['state_dict'])
            model.load_state_dict(model_dict)

    return model
