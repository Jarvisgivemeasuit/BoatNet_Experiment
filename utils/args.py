import torch
import os
class Args:

    def __init__(self):
        self.tr_batch_size = 16
        self.vd_batch_size = 32
        self.num_workers = 8
        self.inplanes = 4

        self.model_name = 'resunet'
        self.backbone = 'resnet50'
        self.backbone1 = 'resnet50'
        self.backbone2 = 'resnet18'
        self.annotations = 'deconv-bilinear'

        self.epochs = 100
        self.lr = 0.01
        self.no_val = False

        self.gpu_ids = [0, 1]
        self.gpu_id = '0, 1'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.vis_image_dir = '/home/arron/Documents/grey/paper/vis_image/'
