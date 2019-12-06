import os
import time
import sys

from collections import namedtuple
from progress.bar import Bar
from apex import amp
from PIL import Image

sys.path.append("/home/arron/Documents/grey/paper/")

from experiment.utils import metrics
from experiment.utils.args import Args
from experiment.utils.utils import *
from experiment.model.boat_resunet.boat_resunet import *
from experiment.model import get_model, save_model
from experiment.dataset.rssrai2 import Rssrai

import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim
import numpy as np


class Trainer:
    def __init__(self, Args):
        self.args = Args
        self.start_epoch = 1
        self.epochs = self.args.epochs
        self.best_pred = 0
        self.best_miou = 0

        train_set, val_set, self.num_classes = make_dataset()
        self.mean = train_set.mean
        self.std = train_set.std
        self.train_loader = DataLoader(train_set, batch_size=self.args.tr_batch_size,
                                       shuffle=True, num_workers=self.args.num_workers)
        self.val_loader = DataLoader(val_set, batch_size=self.args.vd_batch_size,
                                     shuffle=False, num_workers=self.args.num_workers)

        self.net = Boat_UNet_Part1(self.args.backbone, self.args.inplanes).cuda()
        self.conv = nn.Conv2d(512, 2, 1).cuda()
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr)

        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')
        self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        self.criterion0 = SoftCrossEntropyLoss().cuda()
        # self.criterion0 = nn.MSELoss().cuda()
        self.criterion1 = SoftCrossEntropyLoss().cuda()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=4)

        self.Metric = namedtuple('Metric', 'pixacc miou kappa')
        self.train_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        miou=metrics.MeanIoU(self.num_classes),
                                        kappa=metrics.Kappa(self.num_classes))

        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        miou=metrics.MeanIoU(self.num_classes),
                                        kappa=metrics.Kappa(self.num_classes))

    def training(self, epoch):

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        self.net.train()

        for idx, sample in enumerate(self.train_loader):
            img, tar, bmask, rate = sample['image'], sample['label'], sample['binary_mask'], sample['ratios']
            if self.args.cuda:
                img, tar, bmask, rate = img.cuda(), tar.cuda(), bmask.cuda(), rate.cuda()

            self.optimizer.zero_grad()
            pred_rate = self.net(img)[-1]
            pred_rate = self.conv(pred_rate)
            pred_rate = self.pool(pred_rate)
            pred_rate = pred_rate.reshape(pred_rate.shape[0], 2)

            loss = self.criterion0(pred_rate, rate.float())
            # loss = abs((pred_rate - rate.float())).sum() / self.args.tr_batch_size

            losses.update(loss)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.2f}s, Total:{total:}, ETA:{eta:}, Loss:{loss1:.4f}'.format(
                batch=idx + 1,
                size=num_train,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss1=losses.avg,
            )
            
            if idx == 440:
                print(pred_rate[0].tolist(), rate[0].tolist())

            bar.next()
        bar.finish()

        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)

    # def validation(self, epoch):

    #     batch_time = AverageMeter()
    #     losses = AverageMeter()
    #     starttime = time.time()

    #     num_val = len(self.val_loader)
    #     bar = Bar('Validation', max=num_val)

    #     self.net.eval()

    #     for idx, sample in enumerate(self.val_loader):
    #         img, tar, bmask, rate = sample['image'], sample['label'], sample['binary_mask'], sample['rate']
    #         if self.args.cuda:
    #             img, tar, bmask, rate = img.cuda(), tar.cuda(), bmask.cuda(), rate.cuda()

    #         with torch.no_grad():
    #             pred_bmask, pred_rate, pred = self.net(img)

    #         loss = self.criterion0(pred_rate, rate)
    #         losses.update(loss)

    #         batch_time.update(time.time() - starttime)
    #         starttime = time.time()

    #         bar.suffix = '({batch}/{size})Batch:{bt:.2f}s, Total:{total:}, ETA:{eta:}, Loss:{loss:.4f}'.format(
    #             batch=idx + 1,
    #             size=len(self.val_loader),
    #             bt=batch_time.avg,
    #             total=bar.elapsed_td,
    #             eta=bar.eta_td,
    #             loss=losses.avg,
    #         )
    #         bar.next()
    #     bar.finish()

    #     print(f"[Epoch: {epoch}, numImages: {num_val * self.args.vd_batch_size}]")
    #     print(f'Valid Loss: {losses.avg:.4f}')

    #     return new_pred


def train():
    args = Args()
    trainer = Trainer(args)

    print("==> Start training")
    print('Total Epoches:', trainer.epochs)
    print('Starting Epoch:', trainer.start_epoch)
    for epoch in range(trainer.start_epoch, trainer.epochs):
        trainer.training(epoch)
        # if not args.no_val:
        #     new_pred = trainer.validation(epoch)
        #     trainer.scheduler.step(new_pred)
            # trainer.auto_reset_learning_rate()


train()
