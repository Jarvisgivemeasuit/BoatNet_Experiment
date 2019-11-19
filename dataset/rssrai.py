import os
from glob import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import random

from libtiff import TIFF
from PIL import Image
import albumentations as A

from .path import Path
from .rssrai_utils import *


class Rssrai(Dataset):
    NUM_CLASSES = 16
    
    def __init__(self, mode='train', batch_size=256, base_dir=Path.db_root_dir('rssrai')):
        assert mode in ['train', 'val', 'test']
        super().__init__()

        self._mode = mode
        self._batch_size = batch_size
        self._base_dir = base_dir
        self.mean = mean
        self.std = std

        if self._mode == 'train':
            self._tr_dir = os.path.join(self._base_dir, 'train_numpy_256')
            self._tr_data_list = os.listdir(self._tr_dir)
            self.len = len(self._tr_data_list)

        if self._mode == 'val':
            self._image_dir = os.path.join(self._base_dir, 'split_val_256', 'img')
            self._label_dir = os.path.join(self._base_dir, 'split_val_256', 'label')
            self._label_data_list = glob(os.path.join(self._label_dir, '*.tif'))
            self._label_name_list = [name.split('/')[-1] for name in self._label_data_list]
            self.len = len(self._label_name_list)

        if self._mode == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test/test_split/')
            self._test_img_list = os.listdir(self._image_dir)
            self._test_name_list = [name.split('.')[0] for name in self._test_img_list]
            self.len = len(self._test_name_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self._mode == "train":
            return self.load_numpy(idx)
        elif self._mode == 'val':
            return self.load_img(idx)
        else:
            return self.load_test_numpy(idx)

    def load_test_numpy(self, idx):
        img = np.load(os.path.join(self._image_dir, self._test_img_list[idx]))
        return img, self._test_img_list[idx]

    def load_numpy(self, idx):
        sample = np.load(os.path.join(self._tr_dir, self._tr_data_list[idx]))
        sample = {'image': torch.from_numpy(sample['image']), "label": torch.from_numpy(sample['label']).long()}
        return sample

    def load_img(self, idx):
        sample = self._read_data(self._label_name_list[idx])
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