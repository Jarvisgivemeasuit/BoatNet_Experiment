import os
import torch
from model.resunet.resunet import UNet
# from model.boat_resunet.boat_resunet import Boat_UNet


def get_model(model_name, backbone, inplanes, num_classes):
    if model_name == 'resunet':
        return UNet(inplanes, num_classes, backbone[0])


def save_model(model1, model2=None, model_name, backbone1, backbone2=None):
    save_path = '/home/arron/Documents/grey/paper/model_saving/'
    if backbone2 is None or model2 is None:
        torch.save(model, os.path.join(save_path, "{}-{}-bast_pred.pth".format(backbone1, model_name)))
    else:
        torch.save(model1, os.path.join(save_path, "{}part1-{}-bast_pred.pth".format(model_name, backbone1)))
        torch.save(model2, os.path.join(save_path, "{}part2-{}-bast_pred.pth".format(model_name, backbone2)))
    print('saved model successful.')
