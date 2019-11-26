import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('../')

from dataset import rssrai2

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


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

def make_dataset(tr_batch_size, vd_batch_size):
    train_set = rssrai2.Rssrai(mode='train', batch_size=tr_batch_size)
    val_set = rssrai2.Rssrai(mode='val', batch_size=vd_batch_size)
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
            ori_img_name = ori_img_file.split('.')[0]
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
                    
                    if x == 0 and y == 0:
                        img = img[:x34, :y34, :]
                        res[:x34, :y34, :] = img
                    elif x == 0 and y == max_y - 1:
                        img = img[:x34, y14:, :]
                        res[:x34, y * y12 + y14:, :] = img
                    elif x == max_x - 1 and y == 0:
                        img = img[x14:, :y34, :]
                        res[x * x12 + x14:, :y34, :] = img
                    elif x == max_x - 1 and y == max_y - 1:
                        img = img[x14:, y14:, :]
                        res[x * x12 + x14:, y * y12 + y14:, :] = img
                    elif x == 0:
                        img = img[:x34, y14:y34, :]
                        res[:x34, y * y12 + y14:(y + 1) * y12 + y14, :] = img
                    elif y == 0:
                        img = img[x14:x34, :y34, :]
                        res[x * x12 + x14:(x + 1) * x12 + x14, :y34, :] = img
                    elif x == max_x - 1:
                        img = img[x14:, y14:y34, :]
                        res[x * x12 + x14:, y * y12 + y14:(y + 1) * y12 + y14, :] = img
                    elif y == max_y - 1:
                        img = img[x14:x34, y14:, :]
                        res[x * x12 + x14:(x + 1) * x12 + x14, y * y12 + y14:, :] = img
                    else:
                        img = img[x14:x34, y14:y34, :]
                        res[x * x12 + x14:(x + 1) * x12 + x14, y * y12 + y14:(y + 1) * y12 + y14, :] = img
                        
            res_img = Image.fromarray(np.uint8(res))

            final_path = make_sure_path_exists(os.path.join(self.save_path, self._dir))
            res_img.save(os.path.join(final_path, ori_img_file))

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
# class SoftmaxCrossEntropyLoss(nn.Module):
#     def __init__(self, ignore_index=-1):
#         super().__init__()
#         self.ignore_index = ignore_index
        
#     def forward(self, pred, target):
#         target = target.reshape(target.shape[0], 1, target.shape[1], target.shape[2]).double()
#         pred = F.log_softmax(pred, dim=1)
        
#         mask0 = torch.zeros(target.shape).double().cuda()
#         mask1 = torch.ones(target.shape).double().cuda()
        
#         mask = torch.where(target!=self.ignore_index, mask1, mask0)
#         mask = Variable(mask)
#         # mask = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])
        
#         # loss = -(pred * (mask * target).float()).sum(1)
#         loss = -pred.sum(1)
        
#         # return loss / pred.shape[0]
#         return loss
    
    
# class BoatLoss(nn.Module):
#     def __init__(self, ignore_index=-1):
#         super().__init__()
#         self.loss1 = torch.nn.MSELoss()
#         self.loss2 = torch.nn.CrossEntropyLoss()
#         # self.loss3 = SoftmaxCrossEntropyLoss(ignore_index)
    
#     def forward(self, output_bmask, bmask, output_rate, rate):
#         rate_loss = self.loss1(output_rate, rate)
#         binary_loss = self.loss2(output_bmask, bmask)

#         # binary_loss = self.loss3(output_bmask, bmask.long())
#         # output_loss = self.loss2(output_mask, target)
        
#         return rate_loss + binary_loss



if __name__ == '__main__':
    path = '/home/arron/dataset/rssrai2019/test/test_img'
    save_path = '/home/arron/Documents/grey/paper/performs' 
    res_path = '/home/arron/Documents/grey/paper/rssrai_results/resunet-resnet50'
    supermerger = SuperMerger(path, res_path, save_path)
    supermerger.merge_image()
    # print(supermerger.ori_list)