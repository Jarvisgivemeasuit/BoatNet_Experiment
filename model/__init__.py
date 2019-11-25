import os
import torch
from model.resunet.resunet import UNet
from model.boat_resunet.boat_resunet import Boat_UNet


def get_model(model_name, backbone, inplanes, num_classes):
    if model_name == 'resunet':
        return UNet(inplanes, num_classes, backbone[0])

    elif model_name == 'boat_resunet':
        return Boat_UNet(inplanes, num_classes, backbone[0], backbone[1])


def save_model(model, model_name, backbone1, backbone2=None):
    save_path = '/home/arron/Documents/grey/paper/model_saving/'
    if backbone2 is None:
        torch.save(model, os.path.join(save_path, "{}-{}-bast_pred.pth".format(backbone, model_name)))
    else:
        torch.save(model, os.path.join(save_path, "{}-{}-{}-bast_pred.pth".format(backbone1, backbone2, model_name)))
        
    print('saved model successful.')
