import os
import torch
from model.resunet.resunet import UNet

def get_model(model_name, backbone, inplanes, num_classes):
    if model_name == 'resunet':
        return UNet(inplanes, num_classes, backbone)

def save_model(model, model_name, backbone):
    save_path = '/home/arron/Documents/grey/paper/model_saving/'
    torch.save(model, os.path.join(save_path, "{}-{}-bast_pred.pth".format(backbone, model_name)))
    print('saved model successful.')