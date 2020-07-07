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
import utils.metrics as metrics

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
import numpy as np 
from thop import profile, clever_format


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.load('/home/grey/datasets/rssrai/results/pspnet-none-2convs/pspnet-resnet50_True_False.pth')

    def forward(self, x):
        ori_x = x
        size = (x.shape[2], x.shape[3])
        x, x_ = self.model.backbone(x)

        ratios = self.model.ratios(x)

        x = self.model.ppm(x)
        x = self.model.ppm_conv(x)
        x = self.model.classes_conv(x)

        x = F.interpolate(x,
                            size=size,
                            mode='bilinear',
                            align_corners=True)

        out = self.model.out_conv(x)

        # x_weights = self.weight_conv(x)
        ratios_ = torch.sigmoid(ratios)
        posi_feat, x_weights = self.model.position_conv(ori_x, out)

        output = out * ratios_ * posi_feat
        ratios = ratios.reshape(x.shape[0], x.shape[1])
        return (output, out, x_weights, ratios)

class Tester:
    def __init__(self, Args, save_path, batch_size, use_threshold):
        self.args = Args()
        self.batch_size = batch_size
        self.use_threshold = use_threshold

        self.test_set = Rssrai(mode='test')
        self.num_classes = self.test_set.NUM_CLASSES
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

        self.net = Model()
        self.criterion1 = torch.nn.CrossEntropyLoss().cuda()
        self.criterion2 = SoftCrossEntropyLoss().cuda()

        # self.save_path = make_sure_path_exists(os.path.join(save_path, "tmp"))
        # self.final_save_path = make_sure_path_exists(os.path.join(save_path, "unet-resnet50"))
        self.final_save_path = make_sure_path_exists(save_path)

        self.Metric = namedtuple('Metric', 'pixacc miou kappa')
        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                miou=metrics.MeanIoU(self.num_classes),
                                kappa=metrics.Kappa(self.num_classes))


    def testing(self):

        self.val_metric.miou.reset()
        self.val_metric.kappa.reset()
        self.val_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses3 = AverageMeter()
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
                    [output, output_, x_weights, output_ratios] = self.net(img)
                    loss1 = self.criterion1(output, tar.long())
                    loss2 = self.criterion1(output_, tar.long())
                    loss3 = self.criterion2(output_ratios, ratios.float())
                    loss = loss1 + loss2 + loss3
                    losses1.update(loss1)
                    losses2.update(loss2)
                    losses3.update(loss3)
                    losses.update(loss)

                    # output_tmp = F.softmax(output, dim=1)
                    # output_tmp = output.permute(2, 3, 0, 1)
                    # output_ratios = F.softmax(output_ratios, dim=1)
                    # dynamic = output_tmp > (1 - output_ratios) / (self.num_classes - 1)
                    # dynamic = dynamic.permute(2, 3, 0, 1)
                    # output_tmp = output_tmp.permute(2, 3, 0, 1)
                    # output = output_tmp * dynamic.float()
                else:
                    output = self.net(img)
                    loss = self.criterion1(output, tar.long())
                    losses.update(loss)

                # print()
                # ratios = F.softmax(output_ratios, dim=1)
                # atten = (x_weights.expand(output.shape).permute(2, 3, 0, 1) * ratios).permute(2, 3, 0, 1)
                # print(output_[output_ > 0].mean())
                # atten = atten / atten.std() * output_.std()
                # print(output_[0, :, 0, 0], atten[0, :, 0, 0])
            # output = (output - output_) * (1 / output_ratios.reshape(1, 16, 1, 1))
            # print(output.dtype)

            self.val_metric.pixacc.update(output, tar)
            self.val_metric.miou.update(output, tar)
            self.val_metric.kappa.update(output, tar)
            if self.use_threshold:
                self.save_image(output, img_file, x_weights)
            else:
                self.save_image(output, img_file)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f},loss1:{loss1:.4f},loss2:{loss2:.4f},loss3:{loss3:.4f} | Acc:{Acc:.4f} | mIoU:{mIoU: .4f},maxIoU:{maxIoU:.4f},idx:{index1},minIoU:{minIoU:.4f},idx:{index2} | kappa: {kappa: .4f}'.format(
                batch=idx + 1,
                size=len(self.test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss1=losses1.avg,
                loss2=losses2.avg,
                loss3=losses3.avg,
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

        print(self.val_metric.miou.get_all())

    def save_image(self, output, img_file, x_weights=None):
        output = torch.argmax(output, dim=1).cpu().numpy()
        output_rgb_tmp = decode_segmap(output[0], self.num_classes).astype(np.uint8)
        output_rgb_tmp =Image.fromarray(output_rgb_tmp)
        output_rgb_tmp.save(os.path.join(self.final_save_path, img_file[0].replace('npy', 'tif')))
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
    save_result_path = '/home/grey/datasets/rssrai/results/pspnet-none-2convs'
    tester = Tester(Args, save_result_path, 1, use_threshold=True)

    # print("==> Start testing")
    tester.testing()

# test()
def count():
    # model = torch.load('/home/grey/datasets/rssrai/results/pspnet-none-2convs/pspnet-resnet50_True_False.pth')
    model = torch.load('/home/grey/datasets/rssrai/results/deeplab-base/deeplab-resnet50_False_False.pth')

    inputs = torch.randn(1, 4, 256, 256).cuda()
    flops, params = profile(model, inputs=(inputs,), verbose=True)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)

count()
