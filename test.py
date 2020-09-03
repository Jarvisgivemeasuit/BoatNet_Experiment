import os
import time
import sys

from progress.bar import Bar
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple

sys.path.append("/home/arron/Documents/grey/paper/experiment")

from utils.args import Args
from utils.utils import *
from model import get_model, save_model
from dataset.rssrai import Rssrai
from dataset.gid import GID
import utils.metrics as metrics

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
import numpy as np 


class Tester:
    def __init__(self, Args, param_path, save_path, batch_size, use_threshold):
        self.args = Args()
        self.batch_size = batch_size
        self.use_threshold = use_threshold

        self.test_set = Rssrai(mode='test')
        # self.test_set = GID(mode='val')
        self.num_classes = self.test_set.NUM_CLASSES
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

        self.net = torch.load(param_path)
        self.criterion1 = torch.nn.CrossEntropyLoss().cuda()
        self.criterion2 = SoftCrossEntropyLoss().cuda()

        # self.save_path = make_sure_path_exists(os.path.join(save_path, "tmp"))
        # self.final_save_path = make_sure_path_exists(os.path.join(save_path, "unet-resnet50"))
        self.final_save_path = make_sure_path_exists(save_path)

        self.Metric = namedtuple('Metric', 'pixacc miou kappa F1')
        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                miou=metrics.MeanIoU(self.num_classes),
                                kappa=metrics.Kappa(self.num_classes),
                                F1=metrics.F1())
        self.actived = torch.zeros(16, 16)

    def testing(self):

        self.val_metric.miou.reset()
        # self.val_metric.kappa.reset()
        self.val_metric.pixacc.reset()
        self.val_metric.F1.reset()

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_val = len(self.test_loader)
        bar = Bar('Testing', max=num_val)

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval()

        for idx, sample in enumerate(self.test_loader):
            img, tar, img_file = sample['image'], sample['label'], sample['file']

            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()
            with torch.no_grad():
                if self.use_threshold:
                    [output, output_, x_weights, output_ratios] = self.net(img)
                    loss1 = self.criterion1(output, tar.long())
                    loss2 = self.criterion1(output_, tar.long())
                    loss = loss1 + loss2
                    losses1.update(loss1)
                    losses2.update(loss2)
                    losses.update(loss)

                else:
                    output = self.net(img)
                    loss = self.criterion1(output, tar.long())
                    losses.update(loss)

            self.val_metric.pixacc.update(output, tar)
            self.val_metric.miou.update(output, tar)
            # self.val_metric.kappa.update(output, tar)
            self.val_metric.F1.update(output, tar)
            # if self.use_threshold:
            #     self.save_image(output, img_file, x_weights)
            # else:
            #     self.save_image(output, img_file)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f},loss1:{loss1:.4f},loss2:{loss2:.4f} | Acc:{Acc:.4f} | mIoU:{mIoU: .4f},maxIoU:{maxIoU:.4f},idx:{index1},minIoU:{minIoU:.4f},idx:{index2} | F1: {F1: .4f}'.format(
                batch=idx + 1,
                size=len(self.test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss1=losses1.avg,
                loss2=losses2.avg,
                mIoU=self.val_metric.miou.get()[0],
                maxIoU=self.val_metric.miou.get()[1],
                index1=self.val_metric.miou.get()[2][0],
                minIoU=self.val_metric.miou.get()[3],
                index2=self.val_metric.miou.get()[4][0],
                Acc=self.val_metric.pixacc.get(),
                F1=self.val_metric.F1.get()
            )
            bar.next()
        bar.finish()

        print(f'numImages: {num_val * self.batch_size}]')
        print(f'Test Loss: {losses.avg:.4f}')
        print(self.val_metric.F1.get_all())

    def save_image(self, output, img_file, x_weights=None):
        output = torch.argmax(output, dim=1).cpu().numpy()
        output_rgb_tmp = decode_segmap(output[0], self.num_classes).astype(np.uint8)
        output_rgb_tmp =Image.fromarray(output_rgb_tmp)
        output_rgb_tmp.save(os.path.join(self.final_save_path, img_file[0].replace('.npy', '.tif')))
        if self.use_threshold:
            x_weights = x_weights[0][0].cpu().numpy()
            # x_weights = Image.fromarray(x_weights)
            # plt.figure()
            # plt.imshow(x_weights)
            # plt.savefig(os.path.join(self.final_save_path, img_file[0].replace('.npy', '_ratios.tif')))
            # plt.close('all')
            np.save(os.path.join(self.final_save_path, img_file[0].replace('.npy', '_ratios')), x_weights)
            # x_weights.save(os.path.join(self.final_save_path, img_file[0].replace('.npy', '_ratios.tif')))


def test():
    save_result_path = '/home/grey/datasets/rssrai/results/pspnet-baseline/res100'
    # save_result_path = '/home/grey/datasets/GID15/result/pspnet-dpa'
    # param_path = '/home/grey/Documents/rssrai_model_saving/pspnet-resnet50_16.pth'
    # param_path = '/home/grey/datasets/GID15/result/pspnet-dpa/pspnet-resnet50.pth'
    param_path = '/home/grey/datasets/rssrai/results/senet/senet-resnet50.pth'
    torch.load(param_path)
    tester = Tester(Args, param_path, save_result_path, 1, use_threshold=False)

    print("==> Start testing")
    tester.testing()

test()
