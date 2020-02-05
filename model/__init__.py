import os
import torch
from model.resunet.resunet import UNet
from model.boat_resunet.boat_resunet import Boat_UNet
from model.boat_resunet.dt_resunet import Dy_UNet


def get_model(model_name, backbone, inplanes, num_classes):
    if model_name == 'resunet':
        return UNet(inplanes, num_classes, backbone)
    if model_name == 'boat_resunet':
        return Boat_UNet(inplanes, num_classes, backbone[0], backbone[1])
    if model_name == 'dt_resunet':
        return Dy_UNet(inplanes, num_classes, backbone)


def save_model(model, model_name, backbone1, backbone2=None, annotations=None):
    save_path = '/home/arron/Documents/grey/paper/model_saving/'
    if backbone2 is None:
        torch.save(model, os.path.join(save_path, "{}-{}-{}_bast_pred.pth"
                                        .format(backbone1, model_name, annotations)))
    else:
        torch.save(model, os.path.join(save_path, "{}-{}-{}-{}_bast_pred.pth"
                                       .format(model_name, backbone1, backbone2, annotations)))

    print('saved model successful.')
