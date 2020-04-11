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
            self._image_dir = os.path.join(self._base_dir, 'train_split_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'train_split_256', 'mask')
            self._ratios_dir = os.path.join(self._base_dir, 'train_split_256', 'ratios')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        if self._mode == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val_split_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'val_split_256', 'mask')
            self._ratios_dir = os.path.join(self._base_dir, 'val_split_256', 'ratios')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        if self._mode == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test_split_256', 'img')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self._mode != "test":
            return self.load_numpy(idx, self._mode)
        else:
            return self.load_test_numpy(idx)

    def load_test_numpy(self, idx):
        img = np.load(os.path.join(self._image_dir, self._data_list[idx]))
        img = self._test_enhance(img)
        img = img.transpose((2, 0, 1))
        return img, self._data_list[idx]

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

        return sample

    def load_img(self, idx):
        sample = self._read_data(self._label_file_list[idx])
        sample = self._valid_enhance(sample)
        sample['image'] = np.transpose(np.array(sample['image'],dtype='float32'), (2, 0, 1))
        
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
            A.Normalize(mean=self.mean, std=self.std, p=1)
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample['image'] = sample['image'].transpose((1, 2, 0))
        return compose(**sample)    

    def _train_enhance(self, sample):
        compose = A.Compose([
            # A.ShiftScaleRotate(),
            # A.RGBShift(),
            # A.Blur(),
            # A.GaussNoise(),
            # A.ElasticTransform(),
            # A.Cutout(p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        sample['image'] = sample['image'].transpose((1, 2, 0))
        return compose(**sample)

    def _test_enhance(self, image):
        image = image.transpose(1, 2, 0)
        norm = A.Compose([
            A.Normalize(mean=self.mean, std=self.std, p=1)])
        return norm(image=image)['image']