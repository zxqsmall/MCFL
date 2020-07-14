import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T
import torch
import random
import shutil


class AttrDataset(data.Dataset):

    def __init__(self, split, transform=None, target_transform=None, data_path=None):
        data_path_rap = '/data1/zhengxiaoqiang/par_datasets/RAP/dataset.pkl'
        data_path_pa100k = '/data1/zhengxiaoqiang/par_datasets/PAK100/dataset.pkl'
        data_path_peta = '/data1/zhengxiaoqiang/par_datasets/PETA/peta-release/dataset.pkl'
        dataset_info_rap = pickle.load(open(data_path_rap, 'rb+'))
        dataset_info_pa100k = pickle.load(open(data_path_pa100k, 'rb+'))
        dataset_info_peta = pickle.load(open(data_path_peta, 'rb+'))
        self.split = split


        img_id_rap = dataset_info_rap.image_name
        img_id_pa100k = dataset_info_pa100k.image_name
        img_id_peta = dataset_info_peta.image_name

        attr_label_rap = dataset_info_rap.label
        attr_label_pa100k = dataset_info_pa100k.label
        attr_label_peta = dataset_info_peta.label

        assert split in dataset_info_rap.partition.keys(), f'split {split} is not exist'

        self.dataset = 'Three_dataset'
        self.transform = transform
        self.target_transform = target_transform

        self.root_path_rap = dataset_info_rap.root
        self.root_path_pa100k = dataset_info_pa100k.root
        self.root_path_peta = dataset_info_peta.root


        self.attr_id_rap = dataset_info_rap.attr_name
        self.attr_id_pa100k = dataset_info_pa100k.attr_name
        self.attr_id_peta = dataset_info_peta.attr_name

        self.attr_num_rap = len(self.attr_id_rap)
        self.attr_num_pa100k = len(self.attr_id_pa100k)
        self.attr_num_peta = len(self.attr_id_peta)


        self.img_idx_rap = dataset_info_rap.partition[split]
        self.img_idx_pa100k = dataset_info_pa100k.partition[split]
        self.img_idx_peta = dataset_info_peta.partition[split]


        if isinstance(self.img_idx_rap, list):
            self.img_idx_rap = self.img_idx_rap[0]  # default partition 0
            # self.img_idx_pa100k = self.img_idx_pa100k[0]
            self.img_idx_peta = self.img_idx_peta[0]
        self.img_num_rap = self.img_idx_rap.shape[0]
        self.img_num_pa100k = self.img_idx_pa100k.shape[0]
        self.img_num_peta = self.img_idx_peta.shape[0]
        self.img_id_rap = [img_id_rap[i] for i in self.img_idx_rap]
        self.label_rap = attr_label_rap[self.img_idx_rap]
        self.img_id_pa100k = [img_id_pa100k[i] for i in self.img_idx_pa100k]
        self.label_pa100k = attr_label_pa100k[self.img_idx_pa100k]
        self.img_id_peta = [img_id_peta[i] for i in self.img_idx_peta]
        self.label_peta = attr_label_peta[self.img_idx_peta]
        self.img_id = []
        self.label = []
        self.img_id.extend(self.img_id_rap)
        self.img_id.extend(self.img_id_pa100k)
        self.img_id.extend(self.img_id_peta)
        self.label.extend(self.label_rap)
        self.label.extend(self.label_pa100k)
        self.label.extend(self.label_peta)

    def __getitem__(self, index):
        # print(index)
        imgname, gt_label = self.img_id[index], self.label[index]
        if self.split == 'trainval':
            if index < 33268:
                imgpath = os.path.join(self.root_path_rap, imgname)
            elif 33267 < index < 123268:
                imgpath = os.path.join(self.root_path_pa100k, imgname)
            elif index >= 123268:
                imgpath = os.path.join(self.root_path_peta, imgname)
        else:
            if index < 8317:
                imgpath = os.path.join(self.root_path_rap, imgname)
            elif 8316 < index < 18217:
                imgpath = os.path.join(self.root_path_pa100k, imgname)
            elif index >= 18317:
                imgpath = os.path.join(self.root_path_peta, imgname)

        img = Image.open(imgpath)
        gt_label = gt_label.astype(np.float32)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label

    def __len__(self):
        return len(self.img_id)


class AttrDataset_for_newdataset(data.Dataset):

    def __init__(self, split, transform=None, target_transform=None):

        data_path = '/home/visitor2/baseline-MCFL/dataset.pkl'
        data_path_pa100k = '/data1/zhengxiaoqiang/par_datasets/PAK100/dataset.pkl'
        data_path_rap = '/data1/zhengxiaoqiang/par_datasets/RAP/dataset.pkl'

        dataset_info = pickle.load(open(data_path, 'rb+'))
        dataset_info_pa100k = pickle.load(open(data_path_pa100k, 'rb+'))
        dataset_info_rap = pickle.load(open(data_path_rap, 'rb+'))
        img_idx_pa100k = dataset_info_pa100k.partition[split]
        img_idx_rap = dataset_info_rap.partition[split][0]


        img_id = dataset_info.image_name
        img_id_pa100k = dataset_info_pa100k.image_name
        img_id_rap = dataset_info_rap.image_name
        attr_label = dataset_info.label
        attr_label_pa100k = dataset_info_pa100k.label
        attr_label_rap = dataset_info_rap.label

        img_idx = dataset_info.img_idx
        # assert split in dataset_info.partition.keys(), f'split {split} is not exist'
        if split == 'trainval':
            img_id = [img_id[i] for i in img_idx]
            attr_label = [attr_label[i] for i in img_idx]
            img_id_pa100k = [img_id_pa100k[i] for i in img_idx_pa100k]
            attr_label_pa100k = [attr_label_pa100k[i] for i in img_idx_pa100k]
            img_id_rap = [img_id_rap[i] for i in img_idx_rap]
            attr_label_rap = [attr_label_rap[i] for i in img_idx_rap]
        else:
            img_idx_test = []
            for i in range(20000):
                if i not in img_idx:
                    img_idx_test.append(i)
            img_id = [img_id[i] for i in img_idx_test]
            attr_label = [attr_label[i] for i in img_idx_test]
            # img_id_pa100k = [img_id_pa100k[i] for i in img_idx_pa100k]
            # attr_label_pa100k = [attr_label_pa100k[i].astype(int) for i in img_idx_pa100k]
            img_id_pa100k = []
            attr_label_pa100k = []
            img_id_rap = [img_id_rap[i] for i in img_idx_rap]
            attr_label_rap = [attr_label_rap[i] for i in img_idx_rap]
        attr_label_pa100k = [list(map(int, attr_label_pa100k[i])) for i in range(len(attr_label_pa100k))]
        attr_label_rap = [list(map(int, attr_label_rap[i])) for i in range(len(attr_label_rap))]
        self.dataset = 'new'
        self.transform = transform
        self.target_transform = target_transform

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        # img_id.extend(img_id_pa100k)
        img_id.extend(img_id_rap)
        # attr_label.extend(attr_label_pa100k)
        attr_label.extend(attr_label_rap)
        self.img_id = img_id
        self.label = attr_label

    def __getitem__(self, index):

        imgname, gt_label = self.img_id[index], self.label[index]
        imgpath = imgname
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = torch.tensor(gt_label)

        return img, gt_label

    def __len__(self):
        return len(self.img_id)


# def select_picture():
#     data_path = '/home/zhengxiaoqiang/baseline-FocalLoss-tricks-pa100k/dataset.pkl'
#     save_dir = '/data1/zhengxiaoqiang/par_datasets/new_dataset/Tshirt0/'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     dataset_info = pickle.load(open(data_path, 'rb+'))
#     label = dataset_info.label
#     img_path = dataset_info.image_name
#     for i in range(len(label)):
#         if label[i][2] == 0:
#             shutil.copy(img_path[i], save_dir)
#
#     print('copy finished')
# select_picture()
def get_transform():
    height = 256
    width = 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        random_erase(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    target_transform = T.Compose([
        T.ToTensor()
    ])

    return train_transform, valid_transform, target_transform


def random_erase(p=0.5, area_ratio_range=[0.02, 0.4], min_aspect_ratio=0.5, max_attempt=20):
    sl, sh = area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    def _random_erase(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(max_attempt):
            mask_area = np.random.uniform(sl, sh) * image_area
            aspect_ratio = np.random.uniform(rl, rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image

    return _random_erase
