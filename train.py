import os
import time
import sys
import numpy as np

from collections import namedtuple
from progress.bar import Bar
from apex import amp
from PIL import Image
from pprint import pprint
from tensorboardX import SummaryWriter

# sys.path.append("/home/arron/Documents/grey/paper/")

import utils.metrics as metrics
from utils.args import Args
from utils.utils import *
from model import get_model, save_model
from dataset.rssrai import Rssrai

import torch
from torch import nn
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim
import numpy as np 
import random


class Trainer:
    def __init__(self, Args):
        self.num_classes = Rssrai.NUM_CLASSES
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

        self.net = get_model(self.args.model_name, self.args.backbone, 
                             self.args.inplanes, self.num_classes, 
                             self.args.use_threshold, self.args.use_gcn).cuda()

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')

        # self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        self.criterion1 = nn.CrossEntropyLoss().cuda()
        self.criterion2 = SoftCrossEntropyLoss(times=1).cuda()
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [30, 50, 75, 90], 0.3)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=5e-6)

        self.Metric = namedtuple('Metric', 'pixacc miou kappa')

        self.train_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        miou=metrics.MeanIoU(self.num_classes),
                                        kappa=metrics.Kappa(self.num_classes))

        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        miou=metrics.MeanIoU(self.num_classes),
                                        kappa=metrics.Kappa(self.num_classes))

        # self.writer_acc = SummaryWriter('dt_gcn/acc')
        # self.writer_miou = SummaryWriter('dt_gcn/miou')
        # self.writer_kappa = SummaryWriter('dt_gcn/kappa')

        self.writer_acc = SummaryWriter(f'{self.args.board_dir}/acc')
        self.writer_miou = SummaryWriter(f'{self.args.board_dir}/miou')
        self.writer_kappa = SummaryWriter(f'{self.args.board_dir}/kappa')

        self.writer_acc.add_scalar('train', self.train_metric.pixacc.get(), 0)
        self.writer_miou.add_scalar('train', self.train_metric.miou.get()[0], 0)
        self.writer_kappa.add_scalar('train', self.train_metric.kappa.get(), 0)
        self.writer_acc.add_scalar('val', self.val_metric.pixacc.get(), 0)
        self.writer_miou.add_scalar('val', self.val_metric.miou.get()[0], 0)
        self.writer_kappa.add_scalar('val', self.val_metric.kappa.get(), 0)

    def training(self, epoch):

        self.train_metric.miou.reset()
        self.train_metric.kappa.reset()
        self.train_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        self.net.train()
        self.net.freeze_backbone()
        if epoch == 6:
            self.net.train_backbone()

        for idx, sample in enumerate(self.train_loader):
            img, tar, ratios = sample['image'], sample['label'], sample['ratios']

            if self.args.cuda:
                img, tar, ratios = img.cuda(), tar.cuda(), ratios.cuda()

            self.optimizer.zero_grad()
            if self.args.use_threshold:
                [output, output_ratios] = self.net(img)

                loss1 = self.criterion1(output, tar.long())
                loss2 = self.criterion2(output_ratios, ratios.float())
                loss = loss1 + loss2
                losses1.update(loss1)
                losses2.update(loss2)
                losses.update(loss)
            else:
                output = self.net(img)
                loss = self.criterion1(output, tar.long())
                losses.update(loss)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss = scale_loss.half()
                    scale_loss.backward()
            else:
                loss.backward()

            if self.args.use_threshold:
                output_tmp = F.softmax(output, dim=1)
                output_tmp = output_tmp.permute(2, 3, 0, 1)
                output_ratios = F.softmax(output_ratios, dim=1)
                dynamic = output_tmp > (1 - output_ratios) / (self.num_classes - 1)
                dynamic = dynamic.permute(2, 3, 0, 1)
                output_tmp = output_tmp.permute(2, 3, 0, 1)
                output = output_tmp * dynamic.float()

            self.optimizer.step()

            self.train_metric.pixacc.update(output, tar)
            self.train_metric.miou.update(output, tar)
            self.train_metric.kappa.update(output, tar)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss:{loss:.4f},loss1:{loss1:.4f},loss2:{loss2:.4f} | Acc: {Acc:.4f} | mIoU:{mIoU:.4f},maxIoU:{maxIoU:.4f},idx:{index1},minIoU:{minIoU:.4f},idx:{index2} | kappa: {kappa:.4f}'.format(
                batch=idx + 1,
                size=len(self.train_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss1=losses1.avg,
                loss2=losses2.avg,
                mIoU=self.train_metric.miou.get()[0],
                maxIoU=self.train_metric.miou.get()[1],
                index1=self.train_metric.miou.get()[2][0],
                minIoU=self.train_metric.miou.get()[3],
                index2=self.train_metric.miou.get()[4][0],
                Acc=self.train_metric.pixacc.get(),
                kappa=self.train_metric.kappa.get()
            )
            bar.next()
        bar.finish()
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)

        self.writer_acc.add_scalar('train', self.train_metric.pixacc.get(), epoch)
        self.writer_miou.add_scalar('train', self.train_metric.miou.get()[0], epoch)
        self.writer_kappa.add_scalar('train', self.train_metric.kappa.get(), epoch)
        self.writer_acc.add_scalar('train/train_loss', losses.avg, epoch)

    def validation(self, epoch):

        self.val_metric.miou.reset()
        self.val_metric.kappa.reset()
        self.val_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_val = len(self.val_loader)
        bar = Bar('Validation', max=num_val)

        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, tar, ratios = sample['image'], sample['label'], sample['ratios']

            if self.args.cuda:
                img, tar, ratios = img.cuda(), tar.cuda(), ratios.cuda()
            with torch.no_grad():
                if self.args.use_threshold:
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

            if idx % 5 == 0:
                self.visualize_batch_image(img, tar, output, epoch, idx)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f},loss1: {loss1:.4f},loss2: {loss2:.4f}| Acc: {Acc:.4f} | mIoU: {mIoU: .4f},maxIoU:{maxIoU:.4f},idx:{index1},minIoU:{minIoU:.4f},idx:{index2} | kappa: {kappa: .4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
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
            # if self.args.use_threshold:
            #     if idx + 1 == len(self.val_loader):
            #         ii = random.randint(0, self.args.vd_batch_size / 2 - 1)
            #         print()
            #         pprint(output_ratios[ii])
            #         pprint(ratios[ii])
        bar.finish()

        print(f'Validation:[Epoch: {epoch}, numImages: {num_val * self.args.vd_batch_size}]')
        print(f'Valid Loss: {losses.avg:.4f}')
        if self.val_metric.miou.get()[0] > self.best_miou:
            if  self.val_metric.pixacc.get() > self.best_pred:
                self.best_pred = self.val_metric.pixacc.get()
            self.best_miou = self.val_metric.miou.get()[0]
            save_model(self.net, self.args.model_name, 'resnet50', self.best_pred, self.best_miou,
                        self.args.use_threshold, self.args.use_gcn)
        print("-----best acc:{:.4f}, best miou:{:.4f}-----".format(self.best_pred, self.best_miou))

        self.writer_acc.add_scalar('val', self.val_metric.pixacc.get(), epoch)
        self.writer_miou.add_scalar('val', self.val_metric.miou.get()[0], epoch)
        self.writer_kappa.add_scalar('val', self.val_metric.kappa.get(), epoch)
        self.writer_acc.add_scalar('val/val_loss', losses.avg, epoch)

        if epoch == self.args.epochs:
            self.writer.close()
        return self.val_metric.pixacc.get()


    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def visualize_batch_image(self, image, target, output, epoch, batch_index):
        # image (B,C,H,W) To (B,H,W,C)
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
        image_np *= self.std
        image_np += self.mean
        image_np *= 255.0
        image_np = image_np.astype(np.uint8)
        image_np = image_np[:, :, :, 1:]

        # target (B,H,W)
        target = target.cpu().numpy()

        # output (B,C,H,W) to (B,H,W)
        output = torch.argmax(output, dim=1).cpu().numpy()

        for i in range(min(3, image_np.shape[0])):
            img_tmp = image_np[i]
            img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB")).astype(np.uint8)
            target_rgb_tmp = decode_segmap(target[i], self.num_classes).astype(np.uint8)
            output_rgb_tmp = decode_segmap(output[i], self.num_classes).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(131)
            plt.imshow(img_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(132)
            plt.imshow(target_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(133)
            plt.imshow(output_rgb_tmp, vmin=0, vmax=255)
            save_path = os.path.join(self.args.vis_image_dir, f'epoch_{epoch}')
            make_sure_path_exists(save_path)
            plt.savefig(f"{save_path}/{batch_index}-{i}.jpg")
            plt.close('all')


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
