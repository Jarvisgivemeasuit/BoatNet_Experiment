import os
import random
import shutil
import numpy as np

from collections import OrderedDict
from libtiff import TIFF
from progress.bar import Bar
import albumentations as A
import math


color_name_map = OrderedDict({(200, 0, 0): 'industrial land',
                              (250, 0, 150): 'urban residential',
                              (200, 150, 150): 'rural residential',
                              (250, 150, 150): 'traffic land',
                              (0, 200, 0): 'paddy field',
                              (150, 250, 0): 'irrigated land',
                              (150, 200, 150): 'dry cropland',
                              (200, 0, 200): 'garden plot',
                              (150, 0, 250): 'arbor woodland',
                              (150, 150, 250): 'shrub land',
                              (250, 200, 0): 'natural grassland',
                              (200, 200, 0): 'artificial grassland',
                              (0, 0, 200): 'river',
                              (0, 150, 200): 'lake',
                              (0, 200, 250): 'pond',
                              (0, 0, 0): 'else'})

color_index_map = OrderedDict({(200, 0, 0): 0,
                              (250, 0, 150): 1,
                              (200, 150, 150): 2,
                              (250, 150, 150): 3,
                              (0, 200, 0): 4,
                              (150, 250, 0): 5,
                              (150, 200, 150): 6,
                              (200, 0, 200): 7,
                              (150, 0, 250): 8,
                              (150, 150, 250): 9,
                              (250, 200, 0): 10,
                              (200, 200, 0): 11,
                              (0, 0, 200): 12,
                              (0, 150, 200): 13,
                              (0, 200, 250): 14,
                              (0, 0, 0): 15})

mask_colormap = np.array([[200, 0, 0],
                       [250, 0, 150],
                       [200, 150, 150],
                       [250, 150, 150],
                       [0, 200, 0],
                       [150, 250, 0],
                       [150, 200, 150],
                       [200, 0, 200],
                       [150, 0, 250],
                       [150, 150, 250],
                       [250, 200, 0],
                       [200, 200, 0],
                       [0, 0, 200],
                       [0, 150, 200],
                       [0, 200, 250],
                       [0, 0, 0]])
# mask_colormap = np.array([[0, 200, 0],
#                        [150, 250, 0],
#                        [150, 200, 150],
#                        [200, 0, 200],
#                        [150, 0, 250],
#                        [150, 150, 250],
#                        [250, 200, 0],
#                        [200, 200, 0],
#                        [200, 0, 0],
#                        [250, 0, 150],
#                        [200, 150, 150],
#                        [250, 150, 150],
#                        [0, 0, 200],
#                        [0, 150, 200],
#                        [0, 200, 250],
#                        [0, 0, 0]])

mean = (0.50582256, 0.35019113, 0.3704326, 0.33756565)
std = (0.24422198, 0.23245192, 0.22758058, 0.22708698)
NUM_CLASSES = 16


class Path: # 租借服务器路径
    @staticmethod
    def get_root_path(dataset_name):
        if dataset_name == 'rssrai_grey':
<<<<<<< HEAD
            return '/data/grey/rssrai/'
        elif dataset_name == 'gid15':
            return '/data/grey/GID15/'
=======
            return '/home/grey/datasets/rssrai/'
        elif dataset_name == 'gid15':
            return '/home/grey/datasets/GID15/'
>>>>>>> d4db40731bd3d03a72ad57184f070548a3848905


class ProcessingPath:
    def __init__(self):
        self.root_path = Path.get_root_path('gid15')
        self.paths_dict = {}

    def get_paths_dict(self):
        self.paths_dict['ori_path'] = os.path.join(self.root_path, 'train_CMYK')
        self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256')
        self.paths_dict['train_backup'] = os.path.join(self.root_path, 'backup_CMYK')
        self.paths_dict['test'] = os.path.join(self.root_path, 'test_split')

        return self.paths_dict


class RandomImageSpliter:
    def __init__(self, path_dict, crop_size=(256, 256)):
        self.data_path = path_dict['data_path']
        self.train_path = path_dict['save_path']
        self.backup_path = path_dict['backup_path']
        self.crop_size = crop_size
        self.test_range_list = {'GF2_PMS1__L1A0001680858-MSS1_label.tif': [[1600, 300]],
                    'GF2_PMS2__L1A0001787564-MSS2_label.tif': [[4500, 0], [2500, 1500]],
                    'GF2_PMS1__L1A0001395956-MSS1_label.tif': [[5400, 2200], [3000, 5100]],
                    'GF2_PMS2__L1A0001517494-MSS2_label.tif': [[5800, 1800], [0, 2600], [2000, 0]],
                    'GF2_PMS1__L1A0001118839-MSS1_label.tif': [[500, 2000]],
                    'GF2_PMS1__L1A0001064454-MSS1_label.tif': [[1900, 4600]],
                    'GF2_PMS2__L1A0001471436-MSS2_label.tif': [[2200, 1100]],
                    'GF2_PMS2__L1A0000718813-MSS2_label.tif': [[1400, 1100]],
                                'size':[1000, 1000]}
        self.ori_img_list = {}

    def split_tr_image(self, num_samples):
        i = 17500
        bar = Bar('spliting tr image:', max=num_samples)
        make_sure_path_exists(os.path.join(self.train_path, 'img'))
        make_sure_path_exists(os.path.join(self.train_path, 'label'))

        while True:
            if i <= num_samples * 0.7:
                img, label, information = self.random_crop(mode='train', condition=True)
            else:
                img, label, information = self.random_crop(mode='backup', condition=True)

            if ((label == 0).sum(axis=0) == 3).sum() / (self.crop_size[0] * self.crop_size[1]) > 0.8:
                continue

            if information[0] not in self.ori_img_list:
                self.ori_img_list[information[0]] = np.zeros(information[3][:2])
                self.ori_img_list[information[0]][information[1]:information[1] + self.crop_size[0], information[2]:information[2] + self.crop_size[1]] += 1
            else:
                self.ori_img_list[information[0]][information[1]:information[1]  + self.crop_size[0], information[2]:information[2] + self.crop_size[1]] += 1

            np.save(os.path.join(self.train_path, 'img', f'{i}'), img)
            np.save(os.path.join(self.train_path, 'label', f'{i}'), label)
            bar.suffix = f'{i + 1} / {num_samples}'
            bar.next()
            i += 1

            if i == num_samples + 1:
                break
        bar.finish()
        np.save(os.path.join(self.data_path, 'crop_condition'), self.ori_img_list)

    def random_crop(self, mode, condition=False):
        if mode == 'train':
            img_path = os.path.join(self.data_path, 'img')
            label_path = os.path.join(self.data_path, 'label')
        else:
            img_path = os.path.join(self.backup_path, 'img')
            label_path = os.path.join(self.backup_path, 'label')

        file_list = os.listdir(img_path)
        for file in file_list:
            if file[-3:] != 'tif':
                file_list.remove(file)
        img_file = random.choice(file_list)
        label_file = img_file.replace('.tif', '_label.tif')
        
        img_obj = TIFF.open(os.path.join(img_path, img_file))
        img = img_obj.read_image()
        label_obj = TIFF.open(os.path.join(label_path, label_file))
        label = label_obj.read_image()

        if label_file in self.test_range_list:
            for index in self.test_range_list[label_file]:
                reg_size = self.test_range_list['size']
                while True:
                    topY = np.random.randint(img.shape[0] - self.crop_size[0])
                    leftX = np.random.randint(img.shape[1] - self.crop_size[1])

                    xmin = max(topY, index[0])
                    ymin = max(leftX, index[1])
                    xmax = min(topY + self.crop_size[0], index[0] + reg_size[0])
                    ymax = min(leftX + self.crop_size[1], index[1] + reg_size[1])

                    inter = max(0, (xmax - xmin)) * max(0, (ymax - ymin))
                    if inter == 0:
                        break
                    else:
                        continue
        else:
            topY = np.random.randint(img.shape[0] - self.crop_size[0])
            leftX = np.random.randint(img.shape[1] - self.crop_size[1])
        
        crop_image = img[topY:topY + self.crop_size[0], leftX:leftX + self.crop_size[1], :].transpose((2, 0, 1))
        crop_label = label[topY:topY + self.crop_size[0], leftX:leftX + self.crop_size[1], :].transpose((2, 0, 1))
        return crop_image, crop_label, [img_file, topY, leftX, label.shape]


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# 将label图转为segmap形式
def label_indices(mask):
    # # colormap2label
    colormap2label = np.zeros(256**3)
    for i, colormap in enumerate(mask_colormap):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

    # colormap2mask
    mask = mask.astype('int32')
    idx = (mask[0, :, :] * 256 + mask[1, :, :]) * 256 + mask[2, :, :]
    return colormap2label[idx].astype('int32')


def save_label_map(paths_dict):
    label_list = os.listdir(paths_dict['data_path'])
    for label_file in label_list:
        if label_file[-3:] != 'npy':
            label_list.remove(label_file)
        

    num_labels = len(label_list)
    bar = Bar("Transposing label to segmap: ", max=num_labels)
    make_sure_path_exists(os.path.join(paths_dict['save_path']))

    for i, label_file in enumerate(label_list):
        label = np.load(os.path.join(paths_dict['data_path'], label_file))
        mask = label_indices(label)
        label_file = label_file.replace('_label', '')
        np.save(os.path.join(paths_dict['save_path'], label_file), mask)
        
        bar.suffix = f'{i + 1} / {num_labels}'
        bar.next()
    bar.finish()


# 统计类别数量
def statistic(data_path):
    data_list = os.listdir(os.path.join(data_path, 'mask'))
    num = len(data_list)
    bar = Bar('counting:', max=num)
    res = np.zeros(16)
    for idx, data_file in enumerate(data_list):
        mask = np.load(os.path.join(data_path, 'mask', data_file))
        for i in range(16):
            count = (mask == i).sum()
            res[i] += count
            
        bar.suffix = '{} / {}'.format(idx, num)
        bar.next()
    bar.finish()
    return res


# 将多类别label转换成前背景两类mask
def fore_back(path_dict):
    img_list = os.listdir(path_dict['data_path'])
    num_imgs = len(img_list)
    bar = Bar('Saving binary file:', max=num_imgs)

    for i, mask_file in enumerate(img_list):
        mask = np.load(os.path.join(path_dict['data_path'], mask_file))

        back = (mask == 15).sum()
        rate = (mask.size - back) / mask.size

        binary = np.ones(mask.shape)
        binary[np.where(mask == 15)] = 0

        np.save(os.path.join(path_dict['save_path'], mask_file), {'binary_mask': binary, 'rate': rate})
        
        bar.suffix = f'{i + 1} / {num_imgs}'
        bar.next()
    bar.finish()


def distributing(path):
    img_list = os.listdir(path)
    num_imgs = len(img_list)
    bar = Bar('distributing:', max=num_imgs)
    dis = np.zeros(NUM_CLASSES)

    for i, mask_file in enumerate(img_list):
        mask = np.load(os.path.join(path, mask_file))
        for category in range(NUM_CLASSES):
            dis[category] += (mask == category).sum()
            # print(ratios[category])
        
        bar.suffix = f'{i + 1} / {num_imgs}'
        bar.next()
    bar.finish()
    print(dis)


#  计算所有图片像素的均值并调用std
def mean_std(path):
    img_list = os.listdir(path)
    pixels_num = 0
    value_sum = [0, 0, 0, 0]
    files_num = len(img_list)
    bar = Bar('Calculating mean:', max=files_num)

    i = 0
    for img_file in img_list:
        img = np.load(os.path.join(path, img_file)) /255.0
        pixels_num += img.shape[1] * img.shape[2]
        value_sum += np.sum(img, axis=(1, 2))
        i += 1
        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()

    value_mean = value_sum / pixels_num
    value_std = _std(path, img_list, value_mean, pixels_num)
    return value_mean, value_std


# 计算所有图片的标准差
def _std(path, img_list, mean, pixels_num):
    files_num = len(img_list)
    bar = Bar('Calculating std:', max=files_num)
    value_std = [0, 0, 0, 0]
    i = 0
    for img_file in img_list:
        img = np.load(os.path.join(path, img_file)) / 255.0
        value_std += np.sum((img.transpose((1, 2, 0)) - mean).transpose(2, 0, 1) ** 2, axis=(1, 2))
        i += 1
        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()
    return np.sqrt(value_std / pixels_num)



if __name__ == '__main__':
    paths_obj = ProcessingPath()
    paths_dict = paths_obj.get_paths_dict()

    spliter_paths = {}
    spliter_paths['data_path'] = paths_dict['ori_path']
    spliter_paths['save_path'] = paths_dict['train_split_256']
    spliter_paths['backup_path'] = paths_dict['train_backup']
    spliter_paths['img_format'] = '.tif'

    # spliter = RandomImageSpliter(spliter_paths)
    # spliter.split_tr_image(25000)

    transpose_paths = {}
    transpose_paths['data_path'] = os.path.join(paths_dict['train_split_256'], 'label')
    transpose_paths['save_path'] = os.path.join(paths_dict['train_split_256'], 'mask')

    # transpose_paths['data_path'] = os.path.join(paths_dict['test'], 'label')
    # transpose_paths['save_path'] = os.path.join(paths_dict['test'], 'mask')

    # transpose_paths['data_path'] = os.path.join(paths_dict['test'], 'label')
    # transpose_paths['save_path'] = '/home/grey/datasets/test/rssrai_mask'
    save_label_map(transpose_paths)

    # data_path = os.path.join(paths_dict['train_split_256'], 'img')
    # print(mean_std(data_path))

    # dis_path = os.path.join(paths_dict['train_split_256'], 'mask')
    # distributing(dis_path)

test_list = [['GF2_PMS1__L1A0001395956-MSS1_label.tif', 5400, 2200],
 ['GF2_PMS2__L1A0001787564-MSS2_label.tif', 4500, 0],
 ['GF2_PMS2__L1A0001787564-MSS2_label.tif', 2500, 1500],
 ['GF2_PMS1__L1A0001680858-MSS1_label.tif', 1600, 300],
 ['GF2_PMS2__L1A0001517494-MSS2_label.tif', 5800, 1800],
 ['GF2_PMS2__L1A0001517494-MSS2_label.tif', 0, 2600],
 ['GF2_PMS1__L1A0001118839-MSS1_label.tif', 500, 2000],
 ['GF2_PMS2__L1A0001517494-MSS2_label.tif', 2000, 0],
 ['GF2_PMS1__L1A0001395956-MSS1_label.tif', 3000, 5100],
 ['GF2_PMS1__L1A0001064454-MSS1_label.tif', 1900, 4600],
 ['GF2_PMS2__L1A0001471436-MSS2_label.tif', 2200, 1100],
 ['GF2_PMS2__L1A0000718813-MSS2_label.tif', 1400, 1100]]