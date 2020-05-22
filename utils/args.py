import torch
import os
class Args:

    def __init__(self):
        self.tr_batch_size = 32
        self.vd_batch_size = 1

        self.num_workers = 8
        self.inplanes = 4

        self.use_threshold = True
        self.use_gcn = False
        self.model_name = 'pspnet'
        self.backbone = 'resnet50'

        self.epochs = 100

        self.lr = 0.015
        self.no_val = False

        self.gpu_ids = [0]
        self.gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id
        self.cuda = torch.cuda.is_available()
        self.apex = True

        self.vis_image_dir = '/home/grey/datasets/rssrai/vis_image/'
        self.board_dir = 'pspnet_thres'
        # self.vis_image_dir = '/home/arron/Documents/grey/paper/vis_image/'
