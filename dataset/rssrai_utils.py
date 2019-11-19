import os
from collections import OrderedDict

from progress.bar import Bar
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
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

color_list = np.array([[0, 200, 0],
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


def decode_seg_map_sequence(label_masks):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
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
    return rgb


def encode_segmap(label_image):
    """Encode segmentation label images as pascal classes
    Args:
        label_image (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    label_image = label_image.astype(int)
    label_mask = np.zeros((label_image.shape[0], label_image.shape[1]), dtype=np.int16)
    for ii, label in enumerate(color_list):
        label_mask[np.where(np.all(label_image == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def visualize_batch_image(image, target, output,
                          epoch, batch_index,
                          directory):
    # image (B,C,H,W) To (B,H,W,C)
    image_np = image.cpu().numpy()
    image_np = np.transpose(image_np, axes=[0, 2, 3, 1])
    image_np *= std
    image_np += mean
    image_np *= 255.0
    image_np = image_np.astype(np.uint8)

    # target (B,H,W)
    target = target.cpu().numpy()

    # output (B,C,H,W) to (B,H,W)
    output = torch.argmax(output, dim=1).cpu().numpy()

    for i in range(min(3, image_np.shape[0])):
        img_tmp = image_np[i]
        img_rgb_tmp = np.array(Image.fromarray(img_tmp).convert("RGB")).astype(np.uint8)
        target_rgb_tmp = decode_segmap(target[i]).astype(np.uint8)
        output_rgb_tmp = decode_segmap(output[i]).astype(np.uint8)
        plt.figure()
        plt.title('display')
        plt.subplot(131)
        plt.imshow(img_rgb_tmp, vmin=0, vmax=255)
        plt.subplot(132)
        plt.imshow(target_rgb_tmp, vmin=0, vmax=255)
        plt.subplot(133)
        plt.imshow(output_rgb_tmp, vmin=0, vmax=255)

        path = os.path.join(directory, "train_image", f'epoch_{epoch}')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f"{path}/{batch_index}-{i}.jpg")
        plt.close('all')


# 切分原始图片成若干小图片, 50%overlap
class ImageSpliter:
    def __init__(self, path_dict, crop_size=(256, 256)):
        self.data_path = path_dict['data_path']
        self.save_path = path_dict['save_path']
        self.crop_size = crop_size
        self.data_list = []
        self.img_format = path_dict['img_format']

    def get_data_list(self):
        return self.data_list
        
    def split_image(self):
        img_list = os.listdir(self.data_path)
        num_imgs = len(img_list)

        for i, img_file in enumerate(img_list):
            img = np.load(os.path.join(self.data_path, img_file))
            self._img_crop(img, img_file, i)

    def _img_crop(self, img, img_file, i):
        _, height, width = img.shape
        len_y, len_x = self.crop_size
        img_name = img_file.replace(self.img_format, '')

        num_imgs = ((height // len_y + 1) * 2) ** 2
        bar = Bar(f'Image {i + 1} spliting:', max=num_imgs)

        x = 0
        row_count = 0
        while x < height:
            y = 0
            col_count = 0
            while y < width:
                if y >= width - len_y // 2 and x >= height - len_x // 2:
                    split_image = img[:, height - len_x:, width - len_y:]
                elif y >= width - len_y // 2:
                    split_image = img[:, x:x + len_x, width - len_y:]
                elif x >= height - len_x // 2:
                    split_image = img[:, height - len_x:, y:y + len_y]
                else:
                    split_image = img[:, x:x + len_x, y:y + len_y]

                split_image_name = '_'.join([img_name, str(row_count), str(col_count)])
                self.data_list.append(split_image_name)
                np.save(os.path.join(self.save_path, split_image_name), split_image)
                if y == width:
                    break
                
                y = min(width, y + len_y // 2)
                col_count += 1
                bar.suffix = f'{row_count * (height // len_y + 1) * 2 + col_count}/{num_imgs}'
                bar.next()
        
            if x == height:
                break
            x = min(height, x + len_x // 2)
            row_count += 1
        bar.finish()
        print('Image split all complete.')

# 将图片转为numpy数组存储，方便训练时读取
def transpose_test_img_to_numpy(data_path, save_path):
    img_list = os.listdir(data_path)
    bar = Bar('transposing image to numpy\'s array and saving', max=len(img_list))
    for i, img_file in enumerate(img_list):
        img_obj = TIFF.open(os.path.join(data_path, img_file))
        img = img_obj.read_image()

        norm = A.Normalize(mean=mean, std=std, p=1)
        img = norm(image=img)['image']
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))

        np.save(os.path.join(save_path, img_file.split('.')[0]), img)
        bar.suffix = f'{i + 1}/{len(img_list)}'
        bar.next()
    bar.finish()


def fore_back(path_dict):
    img_list = os.listdir(path_dict['data_path'])
    num_imgs = len(img_list)
    bar = Bar('Saving binary file:', max=num_imgs)
    for i, img_file in enumerate(img_list):
        img_np = np.load(os.path.join(path_dict['data_path'], img_file))
        
        back = (img_np['label'] == 15).sum()
        rate = (img_np['label'].size - back) / img_np['label'].size
        
        mask = np.ones(img_np['label'].shape)
        mask[np.where(img_np['label'] == 15)] = 0
        
        save_name = '_'.join([img_file.replace('.npz', ''),'{:.4f}'.format(rate)])
        
        np.savez(os.path.join(path_dict['save_path'], save_name),{'image':img_np['image'], 'label':mask})
        bar.suffix = f'{i + 1} / {num_imgs}'
        bar.next()
    bar.finish()


if __name__ == '__main__':
    path_dict = {}
    # path_dict['data_path'] = '/home/arron/dataset/rssrai2019/test/test_numpy'
    # path_dict['save_path'] = '/home/arron/dataset/rssrai2019/test/test_split'
    # spliter = ImageSpliter(path_dict, (900, 850))
    # spliter.split_image()
    
    
    path_dict['data_path'] = '/home/arron/dataset/rssrai2019/train_numpy_256'
    path_dict['save_path'] = '/home/arron/Documents/grey/paper/binary_label'
    fore_back(path_dict)