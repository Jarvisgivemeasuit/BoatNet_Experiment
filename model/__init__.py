import os
import torch
from model.unet.unet import UNet
from model.pspnet.pspnet import PSPNet
from model.deeplab.deeplab import DeepLabV3Plus
from model.danet.danet import DANet
from model.senet.senet import SENet
from model.CBAM.cbam import CBAM
from model.non_local.non_local import NonLocalNN


def get_model(model_name, backbone, inplanes, num_classes, use_threshold, use_gcn):
    if model_name == 'deeplab':
        return DeepLabV3Plus(inplanes, num_classes, backbone, use_threshold, use_gcn)
    elif model_name == 'pspnet':
        return PSPNet(inplanes, num_classes, backbone, use_threshold, use_gcn)
    elif model_name == 'unet':
        return UNet(inplanes, num_classes, backbone, use_threshold, use_gcn)
    elif model_name == 'danet':
        return DANet(inplanes, num_classes, backbone)
    elif model_name == 'senet':
        return SENet(inplanes, num_classes, backbone)
    elif model_name == 'cbam':
        return CBAM(inplanes, num_classes, backbone)
    elif model_name == 'nonlocal':
        return NonLocalNN(inplanes, num_classes, backbone)


def save_model(model, model_name, backbone, pred, miou, use_threshold, use_gcn):
    save_path = '/home/grey/Documents/rssrai_model_saving/'
    # save_path = '/home/mist/rssrai_model_saving/'
    make_sure_path_exists(save_path)
    # save_path = '/home/arron/Documents/grey/paper/model_saving/'
    # torch.save(model, os.path.join(save_path, "{}-{}-{:.3f}-{:.3f}_{}_{}.pth"
    torch.save(model, os.path.join(save_path, "{}-{}.pth"
                                    .format(model_name, backbone, use_threshold, use_gcn)))

    print('saved model successful.')


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
