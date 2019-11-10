import os
import torch
from model.resunet.resunet import unet

def get_model(model_name, backbone, inplanes, num_classes):
    if model_name == 'resunet':
        return unet(inplanes, num_classes, backbone)

def save_model(model, model_name, backbone, pred, miou):
        save_path = '/home/arron/Documents/grey/paper/experiment/model/model_saving/'
        torch.save(model, os.path.join(save_path, f"{backbone}-{model_name}-acc{pred}-miou{miou}.pth"))
        print('saved model successful.')