import os
import time
import sys

from progress.bar import Bar
from PIL import Image
from collections import namedtuple

sys.path.append("/home/arron/Documents/grey/paper/experiment")

from utils.args import Args
from utils.utils import *
from model import get_model, save_model
from dataset.rssrai2 import Rssrai
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
        self.num_classes = self.test_set.NUM_CLASSES
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.args.num_workers)

        self.net = torch.load(param_path)
        self.criterion1 = torch.nn.CrossEntropyLoss().cuda()
        self.criterion2 = SoftCrossEntropyLoss().cuda()

        self.final_save_path = make_sure_path_exists(os.path.join(save_path, f"{self.args.model_name}-{self.args.backbone}"))
        self.Metric = namedtuple('Metric', 'pixacc miou kappa')
        self.mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
        self.std = (0.24007008, 0.23784, 0.22267079, 0.21865861)
        self.test_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                miou=metrics.MeanIoU(self.num_classes),
                                kappa=metrics.Kappa(self.num_classes))

    def testing(self):
        print('length of test set:', len(self.test_set))

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval()

        num_test = len(self.test_loader)
        bar = Bar('testing', max=num_test)

        for idx, [img, img_file] in enumerate(self.test_loader):
            if self.args.cuda:
                img = img.float().cuda()
            with torch.no_grad():
                if self.use_threshold:
                    [output, output_ratios] = self.net(img)
                else:
                    output = self.net(img)

            # loss1 = self.criterion1(output, tar.long())
            # loss2 = self.criterion2(output_ratios, ratios.float())
            # loss = loss1 + loss2
            # losses1.update(loss1)
            # losses2.update(loss2)
            # losses.update(loss)

            if self.use_threshold:
                output_tmp = F.softmax(output, dim=1)
                output_tmp = output_tmp.permute(2, 3, 0, 1)
                output_ratios = F.softmax(output_ratios, dim=1)
                dynamic = output_tmp > (1 - output_ratios) / (self.num_classes - 1)
                dynamic = dynamic.permute(2, 3, 0, 1)
                output_tmp = output_tmp.permute(2, 3, 0, 1)
                output = output_tmp * dynamic.float()

            # self.test_metric.pixacc.update(output, tar)
            # self.test_metric.miou.update(output, tar)
            # self.test_metric.kappa.update(output, tar)

            self.save_image(output, img_file)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '''({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |
                         '''.format(
                            #  Loss:{loss:.4f},loss1:{loss1:.4f},loss2:{loss2:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f} |'''.format(
                batch=idx + 1,
                size=num_test,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                # loss=losses.avg,
                # loss1=losses1.avg,
                # loss2=losses2.avg,
                # mIoU=self.test_metric.miou.get(),
                # Acc=self.test_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()
        print('testing:')
        print(f"[numImages: {num_test * self.batch_size}] | testing Loss: {losses.avg:.4f}")

    def save_image(self, output, img_file):
        output = torch.argmax(output, dim=1).cpu().numpy()
        for i in range(output.shape[0]):
            output_rgb_tmp = decode_segmap(output[i], self.num_classes).astype(np.uint8)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(self.final_save_path, img_file[i].replace('npy', 'tif')))


def test():
    save_result_path = '/home/arron/dataset/rssrai_grey/results/tmp_output'
    param_path = '/home/arron/Documents/grey/paper/model_saving/dt_resunet-resnet50-0210_bast_pred.pth'
    tester = Tester(Args, param_path, save_result_path, 16, use_threshold=True)

    print("==> Start testing")
    tester.testing()

test()