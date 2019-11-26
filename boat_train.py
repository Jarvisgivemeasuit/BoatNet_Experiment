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
        self.num_classes = Rssrai.NUM_CLASSES
        self.args = Args
        self.start_epoch = 1
        self.epochs = self.args.epochs
        self.best_pred = 0
        self.best_miou = 0

        train_set, val_set, self.num_classes = make_dataset(self.args.tr_batch_size, self.args.vd_batch_size)
        self.mean = train_set.mean
        self.std = train_set.std
        self.train_loader = DataLoader(train_set, batch_size=self.args.tr_batch_size,
                                       shuffle=True, num_workers=self.args.num_workers)
        self.val_loader = DataLoader(val_set, batch_size=self.args.vd_batch_size,
                                     shuffle=False, num_workers=self.args.num_workers)

        self.net_st = Boat_UNet_Part1(self.args.inplanes, 2, self.args.backbone1).cuda()
        self.net_nd = Boat_UNet_Part2(65, self.num_classes, self.args.backbone2).cuda()
        
        self.optimizer = torch.optim.Adam(self.net_st.parameters(), lr=self.args.lr, weight_decay=1e-5)

        if self.args.apex:
            self.net_st, self.optimizer = amp.initialize(self.net_st, self.optimizer, opt_level='O1')
            self.net_nd, self.optimizer = amp.initialize(self.net_nd, self.optimizer, opt_level='O1')
        self.net_st = nn.DataParallel(self.net_st, self.args.gpu_ids)
        self.net_nd = nn.DataParallel(self.net_nd, self.args.gpu_ids)

        self.criterion0 = nn.MSELoss().cuda()
        self.criterion1 = FocalLoss().cuda()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=4)

        self.Metric = namedtuple('Metric', 'pixacc miou kappa')
        self.train_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        miou=metrics.MeanIoU(self.num_classes),
                                        kappa=metrics.Kappa(self.num_classes))

        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                        miou=metrics.MeanIoU(self.num_classes),
                                        kappa=metrics.Kappa(self.num_classes))

    def training(self, epoch):

        self.train_metric.miou.reset()
        self.train_metric.kappa.reset()
        self.train_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        self.net.train()

        for idx, sample in enumerate(self.train_loader):
            img, tar, bmask, rate = sample['image'], sample['label'], sample['binary_mask'], sample['rate']
            if self.args.cuda:
                img, tar, bmask, rate = img.cuda(), tar.cuda(), bmask.cuda(), rate.cuda()

            self.optimizer.zero_grad()
            output_bmask, output_rate, pred_st, down_list, up_list = self.net_st(img)
            loss1 = self.criterion0(output_rate, rate)
            loss2 = self.criterion1(output_bmask, bmask.long())
            
            pred_nd = self.net_nd(pred_st, down_list, up_list)
            loss3 = self.criterion1(pred_nd, tar.long())

            loss = loss1 + loss2 + loss3
            losses.update(loss)

            self.train_metric.pixacc.update(pred_nd, tar)
            self.train_metric.miou.update(pred_nd, tar)
            self.train_metric.kappa.update(pred_nd, tar)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size})Batch:{bt:.2f}s, Total:{total:}, ETA:{eta:}, Loss:{loss1:.4f},{loss2:.4f},{loss3:.4f}, Acc:{Acc:.4f}, mIoU:{mIoU:.4f}, kappa:{kappa:.4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss1=loss1,
                loss2=loss2,
                loss3=loss3,
                mIoU=self.val_metric.miou.get(),
                Acc=self.val_metric.pixacc.get(),
                kappa=self.val_metric.kappa.get()
            )
            bar.next()
        bar.finish()
        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)
        if self.train_metric.pixacc.get() > self.best_pred and self.train_metric.miou.get() > self.best_miou:
            self.best_pred = self.train_metric.pixacc.get()
            self.best_miou = self.train_metric.miou.get()
            save_model(self.net_st, self.net_nd, self.args.model_name, 
                       self.args.backbone1, self.args.backbone2, self.args.annotations)

    def validation(self, epoch):
        self.val_metric.miou.reset()
        self.val_metric.kappa.reset()
        self.val_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_val = len(self.val_loader)
        bar = Bar('Validation', max=num_val)

        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, tar, bmask, rate = sample['image'], sample['label'], sample['binary_mask'], sample['rate']
            if self.args.cuda:
                img, tar, bmask, rate = img.cuda(), tar.cuda(), bmask.cuda(), rate.cuda()

            self.optimizer.zero_grad()
            output_bmask, output_rate, pred_st, down_list = self.net_st(img)

            loss1 = self.criterion0(output_rate, rate)
            loss2 = self.criterion1(output_bmask, bmask.long())
            
            pred_nd = self.net_nd(pred_st, down_list)
            loss3 = self.criterion1(pred_nd, tar.long())
            
            loss = loss1 + loss2 + loss3
            losses.update(loss.item())

            self.val_metric.pixacc.update(pred_nd, tar)
            self.val_metric.miou.update(pred_nd, tar)
            self.val_metric.kappa.update(pred_nd, tar)

            
            if idx % 10 == 0:
                self.visualize_batch_image(img, bmask, output_bmask, epoch, idx, 1)
                self.visualize_batch_image(img, tar, pred_nd, epoch, idx, 2)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size})Batch:{bt:.2f}s, Total:{total:}, ETA:{eta:}, Loss:{loss1:.4f},{loss2:.4f},{loss3:.4f}, Acc:{Acc:.4f}, mIoU:{mIoU:.4f}, kappa:{kappa:.4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss1=loss1,
                loss2=loss2,
                loss3=loss3,
                mIoU=self.val_metric.miou.get(),
                Acc=self.val_metric.pixacc.get(),
                kappa=self.val_metric.kappa.get()
            )
            bar.next()
        bar.finish()

        new_pred = self.val_metric.miou.get()
        metric_str = "Acc:{:.4f}, mIoU:{:.4f}, kappa: {:.4f}".format(self.val_metric.pixacc.get(),
                                                                    new_pred,
                                                                    self.val_metric.kappa.get())
        print('Validation:')
        print(f"[Epoch: {epoch}, numImages: {num_val * self.args.vd_batch_size}]")
        print(f'Valid Loss: {losses.avg:.4f}')

        return new_pred

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def visualize_batch_image(self, image, target, output, epoch, batch_index, stage):
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
            plt.savefig(f"{save_path}/{batch_index}-{i}-stage{stage}.jpg")
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
