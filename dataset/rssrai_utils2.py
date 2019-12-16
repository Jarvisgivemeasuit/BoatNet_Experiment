import os
import random
import shutil
import numpy as np

from collections import OrderedDict
from libtiff import TIFF
from progress.bar import Bar
import albumentations as A


color_name_map = OrderedDict({(0, 200, 0): '水田',
                              (150, 250, 0): '水浇地',
                              (150, 200, 150): '旱耕地',
                              (200, 0, 200): '园地',
                              (150, 0, 250): '乔木林地',
                              (150, 150, 250): '灌木林地',
                              (250, 200, 0): '天然草地',
                              (200, 200, 0): '人工草地',
                              (200, 0, 0): '工业用地',
                              (250, 0, 150): '城市住宅',
                              (200, 150, 150): '村镇住宅',
                              (250, 150, 150): '交通运输',
                              (0, 0, 200): '河流',
                              (0, 150, 200): '湖泊',
                              (0, 200, 250): '坑塘',
                              (0, 0, 0): '其他类别'})

color_index_map = OrderedDict({(0, 200, 0): 0,
                               (150, 250, 0): 1,
                               (150, 200, 150): 2,
                               (200, 0, 200): 3,
                               (150, 0, 250): 4,
                               (150, 150, 250): 5,
                               (250, 200, 0): 6,
                               (200, 200, 0): 7,
                               (200, 0, 0): 8,
                               (250, 0, 150): 9,
                               (200, 150, 150): 10,
                               (250, 150, 150): 11,
                               (0, 0, 200): 12,
                               (0, 150, 200): 13,
                               (0, 200, 250): 14,
                               (0, 0, 0): 15})

mask_colormap = np.array([[0, 200, 0],
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

mean = (0.52891074, 0.38070734, 0.40119018, 0.36884733)
std = (0.24007008, 0.23784, 0.22267079, 0.21865861)
NUM_CLASSES = 16


class Path:
    @staticmethod
    def get_root_path(dataset_name):
        if dataset_name == 'rssrai_grey':
            return '/home/arron/dataset/rssrai_grey/'

        elif dataset_name == 'rssrai_increase':
            return '/home/arron/dataset/rssrai_grey/increase'


class ProcessingPath:
    def __init__(self):
        self.root_path = Path.get_root_path('rssrai_increase')
        self.paths_dict = {}

    def get_paths_dict(self, mode="img"):
        assert mode in ['img', 'label', 'all']

        if mode == "img":
            self.paths_dict['ori_path'] = os.path.join(self.root_path, 'rssrai', 'train', 'img')
            self.paths_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256', 'img')
            self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256', 'img')
            self.paths_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256', 'img')

            self.paths_dict['test_path'] = os.path.join(self.root_path, 'rssrai', 'test')
            self.paths_dict['test_split_256'] = os.path.join(self.root_path, 'test_split_256')

        elif mode == 'label':
            self.paths_dict['ori_path'] = os.path.join(self.root_path, 'rssrai', 'train', 'label')
            self.paths_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256', 'label')
            self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256', 'label')
            self.paths_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256', 'label')

        else:
            self.paths_dict['ori_path'] = os.path.join(self.root_path, 'rssrai', 'train')
            self.paths_dict['data_split_256'] = os.path.join(self.root_path, 'data_split_256')
            self.paths_dict['train_split_256'] = os.path.join(self.root_path, 'train_split_256')
            self.paths_dict['val_split_256'] = os.path.join(self.root_path, 'val_split_256')

            self.paths_dict['test_path'] = os.path.join(self.root_path, 'rssrai', 'test')
            self.paths_dict['test_split_256'] = os.path.join(self.root_path, 'test_split_256')

        return self.paths_dict


# 将原图分割为crop_size的小图
class ImageSpliter:
    def __init__(self, path_dict, crop_size=(256, 256)):
        self.data_path = path_dict['data_path']
        self.save_path = path_dict['save_path']
        self.crop_size = crop_size
        self.img_format = path_dict['img_format']

    def get_data_list(self):
        return self.data_list

    def split_image(self):
        img_list = os.listdir(os.path.join(self.data_path, 'img'))
        num_imgs = len(img_list)

        for i, img_file in enumerate(img_list):
            img_name = img_file.replace(self.img_format, '')
            label_name = "_".join([img_name, 'label'])
            label_file = "".join([label_name, self.img_format])

            img_obj = TIFF.open(os.path.join(self.data_path, 'img', img_file))
            label_obj = TIFF.open(os.path.join(self.data_path, 'label', label_file))
            img, label = img_obj.read_image().transpose((2, 0, 1)), label_obj.read_image().transpose((2, 0, 1))

            self._img_crop(img, img_name, i, 'image')
            self._img_crop(label, label_name, i, 'label')
        print('Sample split all complete.')

    def _img_crop(self, img, img_name, i, tp):
        _, height, width = img.shape
        len_y, len_x = self.crop_size

        num_imgs = (height // len_x + 1) * (width // len_y + 1)
        bar = Bar(f'{tp} {i + 1} spliting:', max=num_imgs)

        x = 0
        row_count = 0
        while x < height:
            y = 0
            col_count = 0
            while y < width:
                if y > width - len_y and x > height - len_x:
                    split_image = img[:, height - len_x:, width - len_y:]
                elif y > width - len_y:
                    split_image = img[:, x:x + len_x, width - len_y:]
                elif x > height - len_x:
                    split_image = img[:, height - len_x:, y:y + len_y]
                else:
                    split_image = img[:, x:x + len_x, y:y + len_y]

                if tp == 'image':
                    split_image_name = '_'.join([img_name, str(row_count), str(col_count)])
                    np.save(os.path.join(self.save_path, 'img', split_image_name), split_image)
                else:
                    split_image_name = '_'.join([img_name.replace('_label', ''), str(row_count), str(col_count)])
                    np.save(os.path.join(self.save_path, 'label', split_image_name), split_image)
                # print(" ", height, width, row_count, col_count)
                if y == width:
                    break
                
                y = min(width, y + len_y)
                col_count += 1
                
                bar.suffix = f'{row_count * (width // len_y + 1) + col_count}/{num_imgs}'
                bar.next()

            if x == height:
                break
            x = min(height, x + len_x)
            row_count += 1
        bar.finish()


class RandomImageSpliter:
    def __init__(self, path_dict, crop_size=(256, 256)):
        self.data_path = path_dict['data_path']
        self.train_path = path_dict['train_path']
        self.val_path = path_dict['val_path']
        self.crop_size = crop_size
        self.valid_range_list = {}

    def split_vd_image(self):
        bar = Bar('spliting vd image:', max=800)
        for i in range(800):
            img, label, information = self.random_crop()
            if information[0] not in self.valid_range_list:
                self.valid_range_list[information[0]] = []
            else:
                self.valid_range_list[information[0]].append(information[1:])

            np.save(os.path.join(self.val_path, 'img', f'{i}'), img)
            np.save(os.path.join(self.val_path, 'label', f'{i}'), label)
            bar.suffix = f'{i + 1} / 800'
            bar.next()
        bar.finish()
        np.save(os.path.join(self.val_path, 'informations'), self.valid_range_list)

    def split_tr_image(self):
        self.valid_range_list = np.load(os.path.join(self.val_path, 'informations.npy'), allow_pickle=True).item()
        i = 0
        bar = Bar('spliting tr image:', max=30000)
        while True:
            img, label, information = self.random_crop()
            ranges = np.array(self.valid_range_list[information[0]]).copy()

            ranges[:, 0][ranges[:, 0] < information[1]] = information[1]
            ranges[:, 1][ranges[:, 1] > information[2]] = information[2]
            x1 = ranges[:, 0]
            y1 = ranges[:, 1]

            ranges = np.array(self.valid_range_list[information[0]]).copy()

            ranges[:, 0] = ranges[:, 0] + self.crop_size[0]
            ranges[:, 1] = ranges[:, 1] + self.crop_size[1]
            ranges[:, 0][ranges[:, 0] > information[1] + self.crop_size[0]] = information[1] + self.crop_size[0]
            ranges[:, 1][ranges[:, 1] > information[2] + self.crop_size[1]] = information[2] + self.crop_size[1]
            x2 = ranges[:, 0]
            y2 = ranges[:, 1]

            if (x1 - x2 > 0).sum() > 0 and (y1 - y2 > 0).sum() > 0:
                continue

            np.save(os.path.join(self.train_path, 'img', f'{i}'), img)
            np.save(os.path.join(self.train_path, 'label', f'{i}'), label)
            bar.suffix = f'{i + 1} / 30000'
            bar.next()
            i += 1
            if i == 30000:
                break
        bar.finish()

    def random_crop(self):
        img_path = os.path.join(self.data_path, 'img')
        label_path = os.path.join(self.data_path, 'label')

        file_list = os.listdir(img_path)
        img_file = random.choice(file_list)
        label_file = img_file.replace('.tif', '_label.tif')

        img_obj = TIFF.open(os.path.join(img_path, img_file))
        img = img_obj.read_image()
        label_obj = TIFF.open(os.path.join(label_path, label_file))
        label = label_obj.read_image()

        topY = np.random.randint(img.shape[0])
        leftX = np.random.randint(img.shape[1])
        
        crop_image = img[topY:topY + self.crop_size[0], leftX:leftX + self.crop_size[1], :]
        crop_label = label[topY:topY + self.crop_size[0], leftX:leftX + self.crop_size[1], :]
        return crop_image, crop_label, [img_file, topY, leftX]


class TestImageSpliter:
    def __init__(self, path_dict, crop_size=(256, 256)):
        self.data_path = path_dict['data_path']
        self.save_path = path_dict['save_path']
        self.crop_size = crop_size
        self.img_format = path_dict['img_format']

    def get_data_list(self):
        return self.data_list

    def split_image(self):
        img_list = os.listdir(os.path.join(self.data_path))

        for i, img_file in enumerate(img_list):
            img_name = img_file.replace(self.img_format, '')

            img_obj = TIFF.open(os.path.join(self.data_path, img_file))
            img = img_obj.read_image().transpose((2, 0, 1))

            self._img_crop(img, img_name, i, 'image')
        print('Sample split all complete.')

    def _img_crop(self, img, img_name, i, tp):
        _, height, width = img.shape
        len_y, len_x = self.crop_size

        num_imgs = (height // (len_x // 2) + 1) * (width // (len_y // 2) + 1)
        bar = Bar(f'{tp} {i + 1} spliting:', max=num_imgs)

        x = 0
        row_count = 0
        while x < height:
            y = 0
            col_count = 0
            while y < width:
                if y > width - len_y // 2 and x > height - len_x // 2:
                    split_image = img[:, height - len_x // 2:, width - len_y // 2:]
                elif y > width - len_y // 2:
                    split_image = img[:, x:x + len_x // 2, width - len_y // 2:]
                elif x > height - len_x // 2:
                    split_image = img[:, height - len_x // 2:, y:y + len_y // 2]
                else:
                    split_image = img[:, x:x + len_x // 2, y:y + len_y // 2]

                if tp == 'image':
                    split_image_name = '_'.join([img_name, str(row_count), str(col_count)])
                    np.save(os.path.join(self.save_path, split_image_name), split_image)

                # print(" ", height, width, row_count, col_count)
                if y == width:
                    break
                
                y = min(width, y + len_y // 2)
                col_count += 1

                bar.suffix = f'{row_count * (width // (len_y // 2) + 1) + col_count}/{num_imgs}'
                bar.next()

            if x == height:
                break
            x = min(height, x + len_x // 2)
            row_count += 1
        bar.finish()


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# 切分训练集和验证集
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
        binary_source = os.path.join(paths_dict['source_path'], 'binary_mask', file)

        if i < num_train:
            img_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'img'))
            label_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'label'))
            mask_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'mask'))
            binary_target = make_sure_path_exists(os.path.join(paths_dict['tr_save_path'], 'binary_mask'))
        else:
            img_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'img'))
            label_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'label'))
            mask_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'mask'))
            binary_target = make_sure_path_exists(os.path.join(paths_dict['vd_save_path'], 'binary_mask'))

        shutil.copy(img_source, img_target)
        shutil.copy(label_source, label_target)
        shutil.copy(mask_source, mask_target)
        shutil.copy(binary_source, binary_target)

        bar.suffix = f'{i + 1}/{num_files}'
        bar.next()
    bar.finish()


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
    num_labels = len(label_list)
    bar = Bar("Transposing label to segmap: ", max=num_labels)
    
    for i, label_file in enumerate(label_list):
        label = np.load(os.path.join(paths_dict['data_path'], label_file))
        mask = label_indices(label)

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


def fore_back_ratios(path_dict):
    img_list = os.listdir(path_dict['data_path'])
    num_imgs = len(img_list)
    bar = Bar('Saving binary file:', max=num_imgs)

    for i, mask_file in enumerate(img_list):
        mask = np.load(os.path.join(path_dict['data_path'], mask_file))

        back = (mask == 15).sum()
        ratios = np.zeros([NUM_CLASSES, 2])
        for category in range(NUM_CLASSES):
            ratios[category, 0] = (mask == category).sum() / mask.size
            ratios[category, 1] = 1 - ratios[category, 0]
            # print(ratios[category])

        binary = np.ones(mask.shape)
        binary[np.where(mask == 15)] = 0

        np.save(os.path.join(path_dict['save_path'], mask_file), {'binary_mask': binary, 'ratios': ratios})
        
        bar.suffix = f'{i + 1} / {num_imgs}'
        bar.next()
    bar.finish()


#  计算所有图片像素的均值并调用std
def mean_std(path):
    img_list = os.listdir(path)
    pixels_num = 0
    value_sum = 0
    files_num = len(img_list)
    bar = Bar('Calculating mean:', max=files_num)

    i = 0
    for img_file in img_list:
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        pixels_num += img.size
        value_sum +=img.sum()
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
    value_std = 0
    i = 0
    for img_file in img_list:
        img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
        value_std += ((img - mean) ** 2).sum()
        i += 1
        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()
    return math.sqrt(value_std / pixels_num)


if __name__ == '__main__':
    paths_obj = ProcessingPath()
    paths_dict = paths_obj.get_paths_dict(mode='all')

    # spliter_paths = {}
    # spliter_paths['data_path'] = paths_dict['test_path']
    # spliter_paths['save_path'] = paths_dict['test_split_256']
    # spliter_paths['img_format'] = '.tif'

    # # spliter = ImageSpliter(spliter_paths)
    # spliter = TestImageSpliter(spliter_paths)
    # spliter.split_image()

    spliter_paths = {}
    spliter_paths['data_path'] = paths_dict['ori_path']
    spliter_paths['train_path'] = paths_dict['train_split_256']
    spliter_paths['val_path'] = paths_dict['val_split_256']

    spliter = RandomImageSpliter(spliter_paths)
    # spliter.split_vd_image()
    spliter.split_tr_image()


    # division_paths = {}
    # division_paths['source_path'] = paths_dict['data_split_256']
    # division_paths['tr_save_path'] = paths_dict['train_split_256']
    # division_paths['vd_save_path'] = paths_dict['val_split_256']

    # train_valid(division_paths)


    # transpose_paths = {}
    # transpose_paths['data_path'] = os.path.join(paths_dict['data_split_256'], 'label')
    # transpose_paths['save_path'] = os.path.join(paths_dict['data_split_256'], 'mask')

    # save_label_map(transpose_paths)


    # binary_paths = {}
    # binary_paths['data_path'] = os.path.join(paths_dict['data_split_256'], 'mask')
    # binary_paths['save_path'] = os.path.join(paths_dict['data_split_256'], 'binary_mask')
    
    # fore_back_ratios(binary_paths)