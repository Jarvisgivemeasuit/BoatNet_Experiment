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
    def __init__(self, Args, param_path, save_path):
        self.args = Args()
        self.test_set = Rssrai(mode='test')
        self.val_set = Rssrai(mode='val')
        self.num_classes = self.test_set.NUM_CLASSES
        self.val_loader = DataLoader(self.val_set, batch_size=2, shuffle=False, num_workers=self.args.num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=2, shuffle=False, num_workers=self.args.num_workers)
        self.net = torch.load(param_path)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.final_save_path = make_sure_path_exists(os.path.join(save_path, f"{self.args.model_name}-{self.args.backbone}"))
        self.Metric = namedtuple('Metric', 'pixacc miou kappa')
        self.val_metric = self.Metric(pixacc=metrics.PixelAccuracy(),
                                miou=metrics.MeanIoU(self.num_classes),
                                kappa=metrics.Kappa(self.num_classes))

    def testing(self, save_path):
        print('length of test set:', len(self.test_set))

        batch_time = AverageMeter()
        starttime = time.time()

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module

        self.net.eval()

        num_test = len(self.test_loader)
        # print(num_test, self.test_set.len)
        bar = Bar('testing', max=num_test)

        for idx, [img, img_file] in enumerate(self.test_loader):
            if self.args.cuda:
                img = img.float().cuda()
            with torch.no_grad():
                output = self.net(img)
            
            
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

    def validation(self):

        self.val_metric.miou.reset()
        self.val_metric.kappa.reset()
        self.val_metric.pixacc.reset()

        batch_time = AverageMeter()
        losses = AverageMeter()
        starttime = time.time()

        num_val = len(self.val_loader)
        bar = Bar('Validation', max=num_val)

        if isinstance(self.net, torch.nn.DataParallel):
            self.net = self.net.module
        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, tar = sample['image'], sample['label']
            # img, tar = sample['image'], sample['binary_mask']
            if self.args.cuda:
                img, tar = img.cuda(), tar.cuda()
            with torch.no_grad():
                output = self.net(img)
            loss = self.criterion(output, tar.long())
            losses.update(loss.item())

            self.val_metric.pixacc.update(output, tar)
            self.val_metric.miou.update(output, tar)
            self.val_metric.kappa.update(output, tar)
            
            if idx % 5 == 0:
                self.visualize_batch_image(img, tar, output, idx)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            # bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f} | kappa: {kappa: .4f}'.format(
            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {Acc: .4f} | mIoU: {mIoU: .4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                mIoU=self.val_metric.miou.get(),
                Acc=self.val_metric.pixacc.get(),
                # kappa=self.val_metric.kappa.get()
            )
            bar.next()
        bar.finish()

        new_pred = self.val_metric.miou.get()
        metric_str = "Acc:{:.4f}, mIoU:{:.4f}, kappa: {:.4f}".format(self.val_metric.pixacc.get(),
                                                                    new_pred,
                                                                    self.val_metric.kappa.get())
        print('Validation:')
        print(f'Valid Loss: {losses.avg:.4f}')

        return new_pred

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def visualize_batch_image(self, image, target, output, batch_index):
        # image (B,C,H,W) To (B,H,W,C)
        # image_np = image.cpu().numpy()
        # image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
        # image_np *= self.std
        # image_np += self.mean
        # image_np *= 255.0
        # image_np = image_np.astype(np.uint8)
        # image_np = image_np[:, :, :, 1:]

        # target (B,H,W)
        target = target.cpu().numpy()

        # output (B,C,H,W) to (B,H,W)
        output = torch.argmax(output, dim=1).cpu().numpy()

        for i in range(min(3, image.shape[0])):
            # img_tmp = image[i]
            # img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB")).astype(np.uint8)
            target_rgb_tmp = decode_segmap(target[i], self.num_classes).astype(np.uint8)
            output_rgb_tmp = decode_segmap(output[i], self.num_classes).astype(np.uint8)
            plt.figure()
            plt.title('display')
            # plt.subplot(131)
            # plt.imshow(img_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(132)
            plt.imshow(target_rgb_tmp, vmin=0, vmax=255)
            plt.subplot(133)
            plt.imshow(output_rgb_tmp, vmin=0, vmax=255)
            output_rgb_tmp =Image.fromarray(output_rgb_tmp)
            output_rgb_tmp.save(os.path.join(self.final_save_path, f'{batch_index}-{i}.tif'))
            # plt.savefig(f"{save_path}/{batch_index}-{i}.jpg")
            plt.close('all')


def test():
    save_result_path = '/home/arron/dataset/rssrai_grey/results/tmp_output'
    param_path = '/home/arron/Documents/grey/paper/model_saving/resunet-resnet50-None_bast_pred.pth'
    tester = Tester(Args, param_path, save_result_path)

    print("==> Start testing")
    tester.validation()

test()