import numpy as np 
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from dataset import rssrai
from tensorboardX import SummaryWriter


def make_dataset(tr_batch_size, vd_batch_size):
    train_set = rssrai.Rssrai(mode='train', batch_size=tr_batch_size)
    val_set = rssrai.Rssrai(mode='val', batch_size=vd_batch_size)
    return train_set, val_set, train_set.NUM_CLASSES
    
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