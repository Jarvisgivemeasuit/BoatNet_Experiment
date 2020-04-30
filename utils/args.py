import torch
import os
class Args:

    def __init__(self):
        self.tr_batch_size = 48
        self.vd_batch_size = 48
        self.num_workers = 8
        self.inplanes = 4

        self.use_threshold = False
        self.use_gcn = False
        self.model_name = 'unet'
        self.backbone = 'resnet50'

        self.epochs = 90
        self.lr = 0.04
        self.no_val = False

        self.gpu_ids = [0]
        self.gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.vis_image_dir = '/home/mist/rssrai/vis_image/'
        self.board_dir = 'unet_base'
        # self.vis_image_dir = '/home/arron/Documents/grey/paper/vis_image/'
