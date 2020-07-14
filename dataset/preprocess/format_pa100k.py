import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

group_order = [7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 1, 2, 3, 0, 4, 5, 6]
# group_order_new = [0, 13, 22, 11, 7, 1]
group_order_new = [0, 13, 22, 9, 10, 11, 7, 1, 24]
# group_order_new = [0, 13, 22, 9, 7, 1, 24]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'release_data/release_data')

    train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
    val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
    test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
    dataset.image_name = train_image_name + val_image_name + test_image_name
    dataset.image_name = [dataset.root + '/' + dataset.image_name[i] for i in range(100000)]
    dataset.label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)
    new_label = np.zeros((100000, 9))
    result_label = np.zeros((100000, 7))
    for step, label in enumerate(dataset.label):
        for i in range(9):
            new_label[step][i] = label[group_order_new[i]]
        if new_label[step][3] == 1 or new_label[step][4] == 1 or new_label[step][5] == 1:
            new_label[step][3] = 1
            result_label[step] = np.delete(new_label[step], [4, 5])


    for label in result_label:
        label[5] = -1


    dataset.label = result_label

    assert dataset.label.shape == (100000, 7)
    dataset.attr_name = [pa100k_data['attributes'][i][0][0] for i in group_order_new]

    # if reorder:
    #     dataset.label = dataset.label[:, np.array(group_order_new)]
    #     dataset.attr_name = [dataset.attr_name[i] for i in group_order_new]

    dataset.partition = EasyDict()
    dataset.partition.train = np.arange(0, 80000)  # np.array(range(80000))
    dataset.partition.val = np.arange(80000, 90000)  # np.array(range(80000, 90000))
    dataset.partition.test = np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.arange(0, 90000)  # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":

    save_dir = '/data1/zhengxiaoqiang/par_datasets/PAK100'
    reoder = True
    generate_data_description(save_dir, reorder=True)
