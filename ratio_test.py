import os
import time
import sys

from collections import namedtuple
from progress.bar import Bar
from apex import amp
from PIL import Image
from pprint import pprint
import random

sys.path.append("/home/arron/Documents/grey/paper/")

from experiment.utils import metrics
from experiment.utils.args import Args
from experiment.utils.utils import *
from experiment.model.resunet.resunet import UNet, RatioNet
from experiment.model.boat_resunet.boat_resunet import *
from experiment.model import get_model, save_model
from experiment.dataset.rssrai2 import Rssrai

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
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

        self.net = RatioNet().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')
        self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        self.criterion = SoftCrossEntropyLoss().cuda()
        # self.criterion = nn.MSELoss().cuda()

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

        self.train_metric.pixacc.reset()
        starttime = time.time()

        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        self.net.train()

        for idx, sample in enumerate(self.train_loader):
            img, ratios = sample['image'], sample['ratios']
            if self.args.cuda:
                img, ratios = img.cuda(), ratios.cuda()

            self.optimizer.zero_grad()
            output = self.net(img)
            output = output.reshape(output.shape[0], output.shape[1])

            loss = self.criterion(output, ratios.float())
            losses.update(loss)
            # self.train_metric.pixacc.update(output, ratios)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.2f}s, Total:{total:}, ETA:{eta:}, Loss:{loss:.4f}, Acc:{Acc:.4f}'.format(
                batch=idx + 1,
                size=num_train,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.train_metric.miou.get(),
                Acc=self.train_metric.pixacc.get(),
            )
            bar.next()
        bar.finish()

        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)

    def validation(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()

        self.val_metric.pixacc.reset()
        starttime = time.time()

        num_valid = len(self.val_loader)
        bar = Bar('Validation', max=num_valid)

        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, ratios = sample['image'], sample['ratios']
            if self.args.cuda:
                img, ratios = img.cuda(), ratios.cuda()

            with torch.no_grad():
                output = self.net(img)
                output = output.reshape(output.shape[0], output.shape[1])

            loss = self.criterion(output, ratios.float())
            losses.update(loss)
            # self.val_metric.pixacc.update(output, ratios)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.2f}s, Total:{total:}, ETA:{eta:}, Loss:{loss:.4f}, Acc:{Acc:.4f}'.format(
                batch=idx + 1,
                size=num_valid,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.val_metric.miou.get(),
                Acc=self.val_metric.pixacc.get(),
            )
            bar.next()
            if idx + 1 == len(self.val_loader):
                index = random.randint(0, 32)
                output = F.softmax(output, dim=1)
                print()
                pprint(output[index])
                pprint(ratios[index])
        bar.finish()

        print('[Epoch: %d, numImages: %5d]' % (epoch, num_valid * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)

        return self.val_metric.pixacc.get()

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
        if not args.no_val:
            new_pred = trainer.validation(epoch)
            trainer.scheduler.step(new_pred)
            # trainer.auto_reset_learning_rate()


train()
