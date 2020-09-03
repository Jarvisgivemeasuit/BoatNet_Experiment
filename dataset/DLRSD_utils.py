import os
import random
import shutil
import numpy as np

from collections import OrderedDict
from libtiff import TIFF
from progress.bar import Bar
import albumentations as A
import math

color_name_map = OrderedDict({(166, 202, 240): 'airplane',
                              (128, 128, 0): 'bare soil',
                              (0, 0, 128): 'buildings',
                              (255, 0, 0): 'cars',
                              (0, 128, 0): 'chaparral',
                              (128, 0, 0): 'court',
                              (255, 233, 233): 'dock',
                              (160, 160, 164): 'field',
                              (0, 128, 128): 'grass',
                              (90, 87, 255): 'mobile home',
                              (255, 255, 0): 'pavement',
                              (255, 192, 0): 'sand',
                              (0, 0, 255): 'sea',
                              (255, 0, 192): 'ship',
                              (128, 0, 128): 'tanks',
                              (0, 255, 0):'trees',
                              (0, 255, 255):'water',
                              (0, 0, 0): 'else'})

color_index_map = OrderedDict({(166, 202, 240): 0,
                              (128, 128, 0): 1,
                              (0, 0, 128): 2,
                              (255, 0, 0): 3,
                              (0, 128, 0): 4,
                              (128, 0, 0): 5,
                              (255, 233, 233): 6,
                              (160, 160, 164): 7,
                              (0, 128, 128): 8,
                              (90, 87, 255): 9,
                              (255, 255, 0): 10,
                              (255, 192, 0): 11,
                              (0, 0, 255): 12,
                              (255, 0, 192): 13,
                              (128, 0, 128): 14,
                              (0, 255, 0): 15,
                              (0, 255, 255): 16,
                              (0, 0, 0): 17})

mask_colormap = np.array([[166, 202, 240],
                              [128, 128, 0],
                              [0, 0, 128],
                              [255, 0, 0],
                              [0, 128, 0],
                              [128, 0, 0],
                              [255, 233, 233],
                              [160, 160, 164],
                              [0, 128, 128],
                              [90, 87, 255],
                              [255, 255, 0],
                              [255, 192, 0],
                              [0, 0, 255],
                              [255, 0, 192],
                              [128, 0, 128],
                              [0, 255, 0],
                              [0, 255, 255],
                              [0, 0, 0]])


class Path:
    @staticmethod
    def get_root_path(dataset_name):
        if dataset_name == 'rssrai_grey':
            return '/home/grey/datasets/rssrai/'


class ProcessingPath:
    def __init__(self):
        self.root_path = Path.get_root_path('rssrai_grey')
        self.paths_dict = {}

    def get_paths_dict(self, mode="img"):
        assert mode in ['img', 'label', 'all']

        if mode == "img":
            self.paths_dict['ori_path'] = os.path.join(self.root_path, 'train', 'img')
            self.paths_dict['data_split_192'] = os.path.join(self.root_path, 'data_split_192', 'img')
            self.paths_dict['train_split_192'] = os.path.join(self.root_path, 'train_split_192', 'img')
            self.paths_dict['val_split_192'] = os.path.join(self.root_path, 'val_split_192', 'img')
            self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256', 'img')
            self.paths_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256', 'img')

            self.paths_dict['test_path'] = os.path.join(self.root_path, 'rssrai', 'test')
            self.paths_dict['test_split_256'] = os.path.join(self.root_path, 'test_split_256')

        elif mode == 'label':
            self.paths_dict['ori_path'] = os.path.join(self.root_path, 'train', 'label')
            self.paths_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256', 'label')
            self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256', 'label')
            self.paths_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256', 'label')

        else:
            self.paths_dict['ori_path'] = os.path.join(self.root_path, 'train')
            self.paths_dict['data_split_192'] = os.path.join(self.root_path, 'data_split_192')
            self.paths_dict['train_split_192'] = os.path.join(self.root_path, 'train_split_192')
            self.paths_dict['val_split_192'] = os.path.join(self.root_path, 'val_split_192')
            self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256')
            self.paths_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256')

            self.paths_dict['test_path'] = os.path.join(self.root_path, 'test_split')
            self.paths_dict['test_split_256'] = os.path.join(self.root_path, 'test_split_256')

        return self.paths_dict


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def train_valid(paths_dict):
    _data_list = os.listdir(os.path.join(paths_dict['source_path'], 'img'))
    num_files = len(_data_list)
    num_train = int(num_files * 0.9)
    num_val = num_files - num_train

    bar = Bar('Dividing trainset and validset:', max=num_files)

    for i in range(num_files):
        file = random.choice(_data_list)
        name = file.split(".")[0]
        _data_list.remove(file)

        img_source = os.path.join(paths_dict['source_path'], 'img', file)
        label_source = os.path.join(paths_dict['source_path'], 'label', file)
        mask_source = os.path.join(paths_dict['source_path'], 'mask', file)
        sta_source = os.path.join(paths_dict['source_path'], 'ratios', file)

        if i < num_train:
            img_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'img'))
            label_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'label'))
            mask_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'mask'))
            sta_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'ratios'))
        else:
            img_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'img'))
            label_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'label'))
            mask_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'mask'))
            sta_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'ratios'))

        shutil.copy(img_source, img_target)
        shutil.copy(label_source, label_target)
        shutil.copy(mask_source, mask_target)
        shutil.copy(sta_source, sta_target)

        bar.suffix = f'{i + 1}/{num_files}'
        bar.next()
    bar.finish()
