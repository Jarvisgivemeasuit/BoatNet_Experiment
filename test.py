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
from dataset.rssrai import Rssrai
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
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

        self.net = torch.load(param_path)
        self.criterion1 = torch.nn.CrossEntropyLoss().cuda()
        self.criterion2 = SoftCrossEntropyLoss().cuda()

        # self.save_path = make_sure_path_exists(os.path.join(save_path, "tmp"))
        self.final_save_path = make_sure_path_exists(os.path.join(save_path, "unet-resnet50"))
        self.Metric = namedtuple('Metric', 'pixacc miou kappa')
        self.mean = (0.49283749, 0.337761, 0.3473801 , 0.33598172)
        self.std = (0.25492469, 0.22505004, 0.20915616, 0.21764152)
        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                miou=metrics.MeanIoU(self.num_classes),
                                kappa=metrics.Kappa(self.num_classes))

    # def testing(self):
    #     print('length of test set:', len(self.test_set))

    #     batch_time = AverageMeter()
    #     losses1 = AverageMeter()
    #     losses2 = AverageMeter()
    #     losses = AverageMeter()
    #     starttime = time.time()

    #     if isinstance(self.net, torch.nn.DataParallel):
    #         self.net = self.net.module

    #     self.net.eval()

    #     num_test = len(self.test_loader)
    #     bar = Bar('testing', max=num_test)

    #     for idx, [img, img_file] in enumerate(self.test_loader):
    #         if self.args.cuda:
    #             img = img.float().cuda()
    #         with torch.no_grad():
    #             if self.use_threshold:
    #                 [output, output_ratios] = self.net(img)
    #             else:
    #                 output = self.net(img)

    #         # loss1 = self.criterion1(output, tar.long())
    #         # loss2 = self.criterion2(output_ratios, ratios.float())
    #         # loss = loss1 + loss2
    #         # losses1.update(loss1)
    #         # losses2.update(loss2)
    #         # losses.update(loss)

    #         if self.use_threshold:
    #             output_tmp = F.softmax(output, dim=1)
    #             output_tmp = output_tmp.permute(2, 3, 0, 1)
    #             output_ratios = F.softmax(output_ratios, dim=1)
    #             dynamic = output_tmp > (1 - output_ratios) / (self.num_classes - 1)
    #             dynamic = dynamic.permute(2, 3, 0, 1)
    #             output_tmp = output_tmp.permute(2, 3, 0, 1)
    #             output = output_tmp * dynamic.float()

    #         # self.test_metric.pixacc.update(output, tar)
    #         # self.test_metric.miou.update(output, tar)
    #         # self.test_metric.kappa.update(output, tar)

    #         self.save_image(output, img_file)

    #         batch_time.update(time.time() - starttime)
    #         starttime = time.time()

    #         bar.suffix = '''({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} |
    #                      '''.format(
    #                         #  Loss:{loss:.4f},loss1:{loss1:.4f},loss2:{loss2:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f} |'''.format(
    #             batch=idx + 1,
    #             size=num_test,
    #             bt=batch_time.avg,
    #             total=bar.elapsed_td,
    #             eta=bar.eta_td,
    #             # loss=losses.avg,
    #             # loss1=losses1.avg,
    #             # loss2=losses2.avg,
    #             # mIoU=self.test_metric.miou.get(),
    #             # Acc=self.test_metric.pixacc.get(),
    #         )
    #         bar.next()
    #     bar.finish()
    #     print('testing:')
    #     print(f"[numImages: {num_test * self.batch_size}] | testing Loss: {losses.avg:.4f}")

    def testing(self):

        self.val_metric.miou.reset()
        self.val_metric.kappa.reset()
        self.val_metric.pixacc.reset()

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
            img, tar, ratios, img_file = sample['image'], sample['label'], sample['ratios'], sample['file']

            if self.args.cuda:
                img, tar, ratios = img.cuda(), tar.cuda(), ratios.cuda()
            with torch.no_grad():
                if self.use_threshold:
                    [output, output_ratios] = self.net(img)

                    loss1 = self.criterion1(output, tar.long())
                    loss2 = self.criterion2(output_ratios, ratios.float())
                    loss = loss1 + loss2
                    losses1.update(loss1)
                    losses2.update(loss2)
                    losses.update(loss)

                    output_tmp = F.softmax(output, dim=1)
                    output_tmp = output.permute(2, 3, 0, 1)
                    output_ratios = F.softmax(output_ratios, dim=1)
                    dynamic = output_tmp > (1 - output_ratios) / (self.num_classes - 1)
                    dynamic = dynamic.permute(2, 3, 0, 1)
                    output_tmp = output_tmp.permute(2, 3, 0, 1)
                    output = output_tmp * dynamic.float()
                else:
                    output = self.net(img)
                    loss = self.criterion1(output, tar.long())
                    losses.update(loss)

            self.val_metric.pixacc.update(output, tar)
            self.val_metric.miou.update(output, tar)
            self.val_metric.kappa.update(output, tar)

            self.save_image(output, img_file)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f},loss1: {loss1:.4f},loss2: {loss2:.4f}| Acc: {Acc:.4f} | mIoU: {mIoU: .4f},maxIoU:{maxIoU:.4f},idx:{index1},minIoU:{minIoU:.4f},idx:{index2} | kappa: {kappa: .4f}'.format(
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
                kappa=self.val_metric.kappa.get()
            )
            bar.next()
        bar.finish()

        print(f'numImages: {num_val * self.batch_size}]')
        print(f'Test Loss: {losses.avg:.4f}')

    def save_image(self, output, img_file):
        output = torch.argmax(output, dim=1).cpu().numpy()
        for i in range(output.shape[0]):
            output_rgb_tmp = decode_segmap(output[i], self.num_classes).astype(np.uint8)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(self.final_save_path, img_file[i].replace('npy', 'tif')))


# def test():
#     save_result_path = '/home/arron/dataset/rssrai_grey/results/tmp_output'
#     param_path = '/home/arron/Documents/grey/paper/model_saving/dt_resunet-resnet50-0210_bast_pred.pth'
#     tester = Tester(Args, param_path, save_result_path, 16, use_threshold=True)

#     print("==> Start testing")
#     tester.testing()

def test():
    save_result_path = '/home/grey/datasets/rssrai/results/'
    param_path = '/home/grey/Documents/rssrai_model_saving/unet-resnet50_False_False.pth'
    torch.load(param_path)
    tester = Tester(Args, param_path, save_result_path, 1, use_threshold=False)

    print("==> Start testing")
    tester.testing()

test()