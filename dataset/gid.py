import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import random

from libtiff import TIFF
from PIL import Image
import albumentations as A

from .gid_utils import *


class GID(Dataset):
    NUM_CLASSES = 16
    
    def __init__(self, mode='train', base_dir=Path.get_root_path('gid15')):
        assert mode in ['train', 'val']
        super().__init__()

        self._mode = mode
        self._base_dir = base_dir
        self.mean = mean
        self.std = std

        if self._mode == 'train':
            self._image_dir = os.path.join(self._base_dir, 'train_split_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'train_split_256', 'mask')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        if self._mode == 'val':
            self._image_dir = os.path.join(self._base_dir, 'test_split', 'img')
            self._label_dir = os.path.join(self._base_dir, 'test_split', 'mask')
            # self._label_dir = '/home/grey/datasets/test/rssrai_mask'
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)
            for data in self._data_list:
                if data[-3:] != 'npy':
                    self._data_list.remove(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.load_numpy(idx, self._mode)

    def load_numpy(self, idx, mode):
        image = np.load(os.path.join(self._image_dir, self._data_list[idx]))
        mask = np.load(os.path.join(self._label_dir, self._data_list[idx]))
        sample = {'image': image, 'label': mask}
        if mode == 'train':
            sample = self._train_enhance(sample)
        else:
            sample = self._test_enhance(sample)

        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['file'] = self._data_list[idx]

        return sample

    def _train_enhance(self, sample):
        compose = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Blur(p=0.5),
            A.Cutout(p=0.5),

            A.Normalize(mean=self.mean, std=self.std, p=1),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample['image'] = sample['image'].transpose((1, 2, 0))
        return compose(**sample)

    def _test_enhance(self, sample):
        # image = image.transpose(1, 2, 0)
        norm = A.Compose([
            A.Normalize(mean=self.mean, std=self.std, p=1)],
            additional_targets={'image': 'image', 'label': 'mask'})
        return norm(**sample)