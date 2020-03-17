import torch
import os
class Args:

    def __init__(self):
        self.tr_batch_size = 32
        self.vd_batch_size = 32
        self.num_workers = 8
        self.inplanes = 4

        self.use_threshold = True
        self.use_gcn = True
        self.model_name = 'unet'
        self.backbone = 'resnet50'
        self.annotations = 'deconv-deconv'

        self.epochs = 150
        self.lr = 0.03
        self.no_val = False

        self.gpu_ids = [0, 1]
        self.gpu_id = '0, 1'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.vis_image_dir = '/home/arron/Documents/grey/paper/vis_image/'
