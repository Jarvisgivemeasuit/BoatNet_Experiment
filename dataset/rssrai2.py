import os
from glob import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import random

from libtiff import TIFF
from PIL import Image
import albumentations as A

# from .path import Path
from .rssrai_utils2 import *


class Rssrai(Dataset):
    NUM_CLASSES = 16
    
    def __init__(self, mode='train', batch_size=256, base_dir=Path.get_root_path('rssrai_grey')):
        assert mode in ['train', 'val', 'test']
        super().__init__()

        self._mode = mode
        self._batch_size = batch_size
        self._base_dir = base_dir
        self.mean = mean
        self.std = std

        if self._mode == 'train':
            self._image_dir = os.path.join(self._base_dir, 'train_split_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'train_split_256', 'mask')
            self._rate_dir = os.path.join(self._base_dir, 'train_split_256', 'binary_mask')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        if self._mode == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val_split_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'val_split_256', 'mask')
            self._rate_dir = os.path.join(self._base_dir, 'val_split_256', 'binary_mask')
            self._data_list = os.listdir(self._image_dir)
            self.len = len(self._data_list)

        # if self._mode == 'test':
        #     self._image_dir = os.path.join(self._base_dir, 'test/test_split/')
        #     self._test_img_list = os.listdir(self._image_dir)
        #     self._test_name_list = [name.split('.')[0] for name in self._test_img_list]
        #     self.len = len(self._test_name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self._mode != "test":
            return self.load_numpy(idx)
        else:
            return self.load_test_numpy(idx)

    def load_test_numpy(self, idx):
        img = np.load(os.path.join(self._image_dir, self._test_img_list[idx]))
        return img, self._test_img_list[idx]

    def load_numpy(self, idx):
        image = np.load(os.path.join(self._image_dir, self._data_list[idx]))
        mask = np.load(os.path.join(self._label_dir, self._data_list[idx]))
        binary_dict = np.load(os.path.join(self._rate_dir, self._data_list[idx]))
        binary_mask, rate = binary_dict['binary_mask'], binary_dict['rate']
        
        sample = {'image': image, 'label': label}
        sample = _train_enhance(sample)
        sample['binary_mask'] = binary_mask
        sample['rate'] = rate
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
        return compose(**sample)    

    def _train_enhance(self, sample):
        compose = A.Compose([
            A.ShiftScaleRotate(),
            A.RGBShift(),
            A.Blur(),
            A.GaussNoise(),
            A.ElasticTransform(),
            A.Cutout(p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        return compose(**sample) 