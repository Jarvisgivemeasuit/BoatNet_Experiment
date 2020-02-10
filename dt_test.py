import os
import time
import sys

from progress.bar import Bar
from PIL import Image

sys.path.append("/home/arron/Documents/grey/paper/experiment")

from utils.args import Args
from utils.utils import *
from model import get_model, save_model
from dataset.rssrai2 import Rssrai

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
import numpy as np 


class Tester:
    def __init__(self, Args):
        self.args = Args()
        self.test_set = Rssrai(mode='test')
        self.num_classes = self.test_set.NUM_CLASSES
        self.test_loader = DataLoader(self.test_set, batch_size=2, shuffle=False, num_workers=self.args.num_workers)
        self.net = None

    def testing(self, param_path, save_path):
        print('length of test set:', len(self.test_set))
        batch_time = AverageMeter()
        starttime = time.time()

        num_test = len(self.test_loader)
        bar = Bar('Testing', max=num_test)
        self.net = torch.load(param_path)
        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval()

        for idx, [img, img_file] in enumerate(self.test_loader):
            if self.args.cuda:
                img = img.float().cuda()
            with torch.no_grad():
                [output, output_ratios] = self.net(img)

            final_save_path = make_sure_path_exists(os.path.join(save_path, f"{self.args.model_name}-{self.args.backbone}"))

            output_tmp = F.softmax(output, dim=1)
            output_tmp = output.permute(2, 3, 0, 1)
            output_ratios = F.softmax(output_ratios, dim=1)
            dynamic = output_tmp > (1 - output_ratios) / (self.num_classes - 1)
            dynamic = dynamic.permute(2, 3, 0, 1)
            output_tmp = output_tmp.permute(2, 3, 0, 1)
            output = output_tmp * dynamic.float()

            output = torch.argmax(output, dim=1).cpu().numpy()
            output_rgb_tmp = decode_segmap(output[0], self.num_classes).astype(np.uint8)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(final_save_path, img_file[0].replace('npy', 'tif')))

            output_rgb_tmp = decode_segmap(output[1], self.num_classes).astype(np.uint8)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(final_save_path, img_file[1].replace('npy', 'tif')))

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                batch=idx + 1,
                size=num_test,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()
        bar.finish()


def test():
    save_result_path = '/home/arron/dataset/rssrai_grey/results/tmp_output'
    param_path = '/home/arron/Documents/grey/paper/model_saving/resnet50-dt_resunet-None_bast_pred.pth'
    # param_path = '/home/arron/Documents/grey/paper/model_saving/resnet50-resunet-None_bast_pred.pth'
    tester = Tester(Args)

    print("==> Start testing")
    tester.testing(param_path, save_result_path)

test()