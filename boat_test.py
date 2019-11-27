import os
import time
import sys

from progress.bar import Bar
from PIL import Image

sys.path.append("/home/arron/Documents/grey/paper/experiment")

from utils.args import Args
from utils.utils import *
from dataset.rssrai2 import Rssrai

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
import numpy as np 


class Tester:
    def __init__(self, Args):
        self.args = Args()
        self.test_set = Rssrai(mode='test')
        self.num_classes = self.test_set.NUM_CLASSES
        self.test_loader = DataLoader(self.test_set, batch_size=10, 
                                      shuffle=False, num_workers=self.args.num_workers)

    def testing(self, param_path, save_path):

        batch_time = AverageMeter()
        starttime = time.time()

        self.net_st = torch.load(param_path[0])
        self.net_nd = torch.load(param_path[1])

        if isinstance(self.net_st, torch.nn.DataParallel):
            self.net_st = self.net_st.module
            self.net_nd = self.net_nd.module

        self.net.eval()

        num_test = len(self.test_loader)
        bar = Bar('testing', max=num_test)

        for idx, [img, img_file] in enumerate(self.test_loader):
            if self.args.cuda:
                img = img.cuda()

            with torch.no_grad():
                output_bmask, output_rate, pred_st, down_list, up_list = self.net_st(img)
                output = self.net_nd(pred_st, down_list)

            final_save_path = make_sure_path_exists(os.path.join(save_path, f"{self.args.model_name}-{self.args.backbone1}-{self.args.backbone2}"))
            output = torch.argmax(output, dim=1).cpu().numpy()
            output_rgb_tmp = decode_segmap(output[0], self.num_classes).astype(np.uint8)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(final_save_path, img_file[0].replace('npy', 'tif')))

            output_rgb_tmp = decode_segmap(output[1], self.num_classes).astype(np.uint8)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(final_save_path, img_file[1].replace('npy', 'tif')))

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size})Batch:{bt:.2f}s, Total:{total:}, ETA:{eta:}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()
        bar.finish()
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))

def test():
    save_result_path = '/home/arron/Documents/grey/paper/rssrai_results'
    param_path1 = '/home/arron/Documents/grey/paper/model_saving/'
    param_path2 = ''
    param_path = [param_path1, param_path2]
    tester = Tester(Args)

    print("==> Start testing")
    tester.testing(param_path, save_result_path)

test()