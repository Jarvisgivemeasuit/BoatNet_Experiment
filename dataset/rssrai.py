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

from .rssrai_utils import *


class Rssrai(Dataset):
    NUM_CLASSES = 16
    
    def __init__(self, mode='train', base_dir=Path.get_root_path('rssrai_grey')):
        assert mode in ['train', 'val', 'test']
        super().__init__()

        self._mode = mode
        self._base_dir = base_dir
        self.mean = mean
        self.std = std

        if self._mode == 'train':
            self._image_dir = os.path.join(self._base_dir, 'train_split_192', 'img')
            self._label_dir = os.path.join(self._base_dir, 'train_split_192', 'mask')
            self._ratios_dir = os.path.join(self._base_dir, 'train_split_192', 'ratios')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        if self._mode == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val_split_192', 'img')
            self._label_dir = os.path.join(self._base_dir, 'val_split_192', 'mask')
            self._ratios_dir = os.path.join(self._base_dir, 'val_split_192', 'ratios')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        if self._mode == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test_split', 'img')
            self._label_dir = os.path.join(self._base_dir, 'test_split', 'mask')
            self._ratios_dir = os.path.join(self._base_dir, 'test_split', 'ratios')
            self._data_list = os.listdir(self._image_dir)
            for data in self._data_list:
                if data[-3:] != 'npy':
                    self._data_list.remove(data)
            self.len = len(self._data_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.load_numpy(idx, self._mode)

    def load_test_numpy(self, idx):
        image = np.load(os.path.join(self._image_dir, self._data_list[idx]))
        mask = np.load(os.path.join(self._label_dir, self._data_list[idx]))
        ratios = np.load(os.path.join(self._ratios_dir, self._data_list[idx]))

        sample = {'image': image, 'label': mask}
        sample = self._test_enhance(sample)

        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['ratios'] = np.array(ratios[:, 0])
        sample['file'] = self._data_list[idx]
        return sample

    def load_numpy(self, idx, mode):
        image = np.load(os.path.join(self._image_dir, self._data_list[idx]))
        mask = np.load(os.path.join(self._label_dir, self._data_list[idx]))
        ratios = np.load(os.path.join(self._ratios_dir, self._data_list[idx]))
        sample = {'image': image, 'label': mask}
        if mode == 'train':
            sample = self._train_enhance(sample)
        else:
            sample = self._valid_enhance(sample)

        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['ratios'] = np.array(ratios[:, 0])
        sample['file'] = self._data_list[idx]

        return sample

    def _read_data(self, label_name):
        image_name = label_name.replace("_label", "")
        image_pil = Image.open(os.path.join(self._image_dir, image_name))
        image_np = np.array(image_pil)

        label_pil = Image.open(os.path.join(self._label_dir, label_name))
        label_np = np.array(label_pil)
        label_mask = encode_segmap(label_np)

        return {'image': image_np, 'label': label_mask}

    def _valid_enhance(self, sample):
        compose = A.Compose([
            # A.Normalize(mean=(0.54010072, 0.40851444, 0.4173501 , 0.38801662), std=(0.54010072, 0.40851444, 0.4173501, 0.38801662), p=1)
            A.Normalize(mean=(0.54299763, 0.37632373, 0.39589563, 0.36152624), std=(0.25004608, 0.24948001, 0.23498456, 0.23068938))
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample['image'] = sample['image'].transpose((1, 2, 0))
        return compose(**sample)    

    def _train_enhance(self, sample):
        compose = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RGBShift(p=0.5),
            A.Blur(p=0.5),
            A.GaussNoise(p=0.5),
            A.ElasticTransform(p=0.5),
            # A.Cutout(p=0.3),
            A.Normalize(mean=self.mean, std=self.std, p=1),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample['image'] = sample['image'].transpose((1, 2, 0))
        return compose(**sample)

    def _test_enhance(self, sample):
        # image = image.transpose(1, 2, 0)
        norm = A.Compose([
            A.Normalize(mean=(0.49283749, 0.337761, 0.3473801 , 0.33598172), std=(0.25492469, 0.22505004, 0.20915616, 0.21764152), p=1)],
            additional_targets={'image': 'image', 'label': 'mask'})
        return norm(**sample)