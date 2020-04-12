import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('../')

from dataset import rssrai

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


NUM_CLASSES = 16
def get_labels(label_number):
    """
    :return: (19 , 3)
    """
    label_19 = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

    label_21 = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]])

    label_16 = np.array([[0, 200, 0],
                         [150, 250, 0],
                         [150, 200, 150],
                         [200, 0, 200],
                         [150, 0, 250],
                         [150, 150, 250],
                         [250, 200, 0],
                         [200, 200, 0],
                         [200, 0, 0],
                         [250, 0, 150],
                         [200, 150, 150],
                         [250, 150, 150],
                         [0, 0, 200],
                         [0, 150, 200],
                         [0, 200, 250],
                         [0, 0, 0]])


    label_colors = {19: label_19, 21: label_21, 16: label_16}
    return label_colors[label_number]


def make_dataset():
    train_set = rssrai.Rssrai(mode='train')
    val_set = rssrai.Rssrai(mode='val')
    return train_set, val_set, train_set.NUM_CLASSES


def decode_segmap(label_mask, label_number):
    """Decode segmentation class labels into a color image
        :param label_mask:
        :param label_number:
    """
    color_list = get_labels(label_number)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(color_list)):
        r[label_mask == ll] = color_list[ll, 0]
        g[label_mask == ll] = color_list[ll, 1]
        b[label_mask == ll] = color_list[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return rgb.astype(np.uint8)


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TensorboardSummary:
    def __init__(self, directory):
        self.directory = directory
        self.writer = SummaryWriter(logdir=os.path.join(self.directory))
        plt.axis('off')


# 将output拼接成完整的图片
class Merger:
    def __init__(self, ori_path, res_path, save_path):
        self.res_path = res_path
        self.save_path = save_path
        self.ori_list, self.height, self.width = get_ori_list_and_size(ori_path) 

    def merge_image(self):
        max_x, max_y = self.find_max_index(self.res_path, self.ori_list)
        for img_file in self.ori_list:
            ori_img_name = img_file.replace('.tif', '')
            res = np.zeros((self.height, self.width, 3))
            for x in range(max_x):
                for y in range(max_y):
                    img_name = '_'.join([ori_img_name, str(x), str(y)])
                    img_file = '.'.join([img_name, 'tif'])
                    img = np.array(Image.open(os.path.join(self.res_path, img_file)))
                    len_x, len_y, _ = img.shape
                    res[x * len_x:x * len_x + len_x, y * len_y:y * len_y + len_y, :] = img
            res_img = Image.fromarray(np.uint8(res))
            res_img.save(os.path.join(self.save_path, img_file))
            print(f"{ori_img_name} merge complete.")
    
    # 找出有多少张output可以组成一张原始图片
    def find_max_index(self):
        img_list = os.listdir(self.res_path)
        xs, ys = [], []

        for img_file in img_list:
            img_name, x, y = self.get_image_message(img_file)
            if self.ori_list[0].replace('.tif', '') == img_name[:-1]:
                xs.append(int(x))
                ys.append(int(y))
        return max(xs), max(ys)
            
    def get_image_message(self, img_file):
        split_tmp = img_file.split('_')[-2:]
        y, x = split_tmp[0], split_tmp[1].replace('.tif', '')
        return img_file.replace("_".join(split_tmp), ''), x, y

# 获得原始test图片的列表和尺寸
def get_ori_list_and_size(path):
    ori_list = os.listdir(path)
    height, width, _ = np.array(Image.open(os.path.join(path, ori_list[0]))).shape
    return ori_list, height, width


# 取output中间1/2height, 1/2width部分拼接成完整图片 
class SuperMerger:
    def __init__(self, ori_path, res_path, save_path):
        self.res_path = res_path
        self._dir = res_path.split("/")[-1]
        self.save_path = save_path
        self.ori_list, self.height, self.width = get_ori_list_and_size(ori_path) 

    def merge_image(self):
        max_x, max_y = self.find_max_index()
        for ori_img_file in self.ori_list:
            ori_img_name = ori_img_file.split('.')[0].strip()
            res = np.zeros((self.height, self.width, 3))
            for x in range(max_x):
                for y in range(max_y):
                    img_name = '_'.join([ori_img_name, str(x), str(y)])
                    img_file = '.'.join([img_name, 'tif'])
                    img = np.array(Image.open(os.path.join(self.res_path, img_file)))

                    len_x, len_y, _ = img.shape
                    x34, y34 = int(len_x * 0.75), int(len_y * 0.75)
                    x14, y14 = int(len_x * 0.25), int(len_y * 0.25)
                    x12, y12 = x34 - x14, y34 - y14
                    
                    # print(x, x * x12 + x14, y, y * y12 + y14)
                    if x == 0 and y == 0:
                        img = img[:x34, :y34, :]
                        res[:x34, :y34, :] = img
                    elif x == 0 and y == max_y - 1:
                        img = img[:x34, y14:, :]
                        res[:x34, -y34:, :] = img
                    elif x == max_x - 1 and y == 0:
                        img = img[x14:, :y34, :]
                        res[-x34:, :y34, :] = img
                    elif x == max_x - 1 and y == max_y - 1:
                        img = img[x14:, y14:, :]
                        res[-x34:,-y34:, :] = img
                    elif x == 0:
                        img = img[:x34, y14:y34, :]
                        res[:x34, y * y12 + y14:(y + 1) * y12 + y14, :] = img
                    elif y == 0:
                        img = img[x14:x34, :y34, :]
                        res[x * x12 + x14:(x + 1) * x12 + x14, :y34, :] = img
                        
                    elif x == max_x - 1:
                        img = img[x14:, y14:y34, :]
                        res[-x34:, y * y12 + y14:(y + 1) * y12 + y14, :] = img
                    elif y == max_y - 1:
                        img = img[x14:x34, y14:, :]
                        res[x * x12 + x14:(x + 1) * x12 + x14, -y34:, :] = img
                    else:
                        img = img[x14:x34, y14:y34, :]
                        res[x * x12 + x14:(x + 1) * x12 + x14, y * y12 + y14:(y + 1) * y12 + y14, :] = img
                        
            res_img = Image.fromarray(np.uint8(res))

            final_path = make_sure_path_exists(self.save_path)
            res_img.save(os.path.join(final_path, ori_img_file))

            print(f"{ori_img_name} merge complete.")

    # 找出有多少张output可以组成一张原始图片
    def find_max_index(self):
        img_list = os.listdir(self.res_path)
        xs, ys = [], []

        for img_file in img_list:
            # print(img_file)
            img_name, x, y = self.get_image_message(img_file)
            if self.ori_list[0].replace(' .tif', '') == img_name[:-1]:
                xs.append(int(x))
                ys.append(int(y))
        return max(xs), max(ys)

    def get_image_message(self, img_file):
        # print(img_file)
        split_tmp = img_file.split('_')[-2:]
        x, y = split_tmp[0], split_tmp[1].replace('.tif', '')
        return img_file.replace("_".join(split_tmp), ''), x, y


class SoftCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-1, times=1, eps=1e-7):
        super().__init__()
        self.ignore_index = ignore_index
        self.times = times
        self.eps = eps

    def forward(self, pred, target):
        mask = target != self.ignore_index
        pred = F.log_softmax(pred, dim=-1)
        loss = -pred * target
        loss = loss * mask.float()
        # print(loss, pred, target, mask)
        return self.times * loss.sum() / (mask.sum() + self.eps)


class Circumference(nn.Module):
    '''计算mask中所有种类的平均周长'''
    def __init__(self):
        super().__init__()

    def cal_circle(self, pred):
        segmap = self.label_indices(pred.transpose(2, 0, 1))
        print(segmap.shape)
        scale = segmap.shape
        ans_rows = np.zeros((scale[0] - 1, scale[1]))
        ans_cols = np.zeros((scale[1] - 1, scale[0]))
        for i in range(scale[0] - 1):
            ans_rows[i] = segmap[i] != segmap[i + 1]
        for i in range(scale[1] - 1):
            ans_cols[i] = segmap[:, i] != segmap[:, i + 1]
        return (ans_rows.sum() + ans_cols.sum()) / NUM_CLASSES

    def label_indices(self, mask):
        # # colormap2label
        colormap2label = np.zeros(256**3)
        mask_colormap = get_labels(NUM_CLASSES)
        for i, colormap in enumerate(mask_colormap):
            colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

        # colormap2mask
        mask = mask.astype('int32')
        idx = (mask[0, :, :] * 256 + mask[1, :, :]) * 256 + mask[2, :, :]
        return colormap2label[idx].astype('int32')


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, eps=1e-7, reducation='mean'):
        super().__init__()
        self.alpha = Variable(torch.tensor(alpha))
        self.gamma = gamma
        self.eps = eps
        self.reducation = reducation

    def forward(self, pred, target):
        N = pred.shape[0]
        C = pred.shape[1]
        num_pixels = pred.shape[2] * pred.shape[3]

        target_index = target.view(target.shape[0], target.shape[1], target.shape[2], 1)
        class_mask = torch.zeros([N, pred.shape[2], pred.shape[3], C]).cuda()
        class_mask = class_mask.scatter_(3, target_index, 1.)
        class_mask = class_mask.transpose(1, 3)
        class_mask = class_mask.view(pred.shape)

        logsoft_pred = F.log_softmax(pred, dim=1)
        soft_pred = F.softmax(pred, dim=1)

        loss = -self.alpha * ((1 - soft_pred)) ** self.gamma * logsoft_pred
        loss = loss * class_mask
        loss = loss.sum(1)

        if self.reducation == 'mean':
            return loss.sum() / (class_mask.sum() + self.eps)
        else:
            return loss.sum()


if __name__ == '__main__':
    # path = '/home/arron/dataset/rssrai_grey/rssrai/train/img'
    # path = '/home/arron/dataset/rssrai_grey/increase/rssrai/test'
    # save_path = '/home/arron/dataset/rssrai_grey/results/dt_resunet-resnet50' 
    # res_path = '/home/arron/dataset/rssrai_grey/results/tmp_output/dt_resunet-resnet50'

    path = '/home/mist/rssrai/ori_img/val/img'
    save_path = '/home/mist/results/unet-resnet50' 
    res_path = '/home/mist/results/tmp'
    supermerger = SuperMerger(path, res_path, save_path)
    supermerger.merge_image()


    # lists = os.listdir(res_path)
    # i = 0
    # for files in lists:
    #     if '_'.join(files.split('_')[:-2]) == 'GF2_PMS1__20160623_L1A0001660727-MSS1':
    #         print(files)
    #         i += 1
    # print(i)
    # print(len(lists))
    