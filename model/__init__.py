import os
import torch
from model.dt_unet.dt_unet import Dt_UNet


def get_model(model_name, backbone, inplanes, num_classes, use_threshold, use_gcn):
    if model_name == 'resunet':
        return UNet(inplanes, num_classes, backbone)
    if model_name == 'boat_resunet':
        return Boat_UNet(inplanes, num_classes, backbone[0], backbone[1])
    if model_name == 'unet':
        return Dt_UNet(inplanes, num_classes, backbone, use_threshold, use_gcn)


def save_model(model, model_name, backbone, pred, miou, use_threshold, use_gcn):
    save_path = '/home/arron/Documents/grey/paper/model_saving/'
    torch.save(model, os.path.join(save_path, "{}-{}-{:.3f}-{:.3f}_{}_{}.pth"
                                    .format(model_name, backbone, pred, miou, use_threshold, use_gcn)))

    print('saved model successful.')
