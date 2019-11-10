import numpy as np 
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from dataset import rssrai
from tensorboardX import SummaryWriter


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
    train_set = rssrai.Rssrai(mode='train', batch_size=tr_batch_size)
    val_set = rssrai.Rssrai(mode='val', batch_size=vd_batch_size)
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