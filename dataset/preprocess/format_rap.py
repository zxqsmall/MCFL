import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat
import torch

np.random.seed(0)
random.seed(0)


# group_order = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
#                26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 1, 2, 3, 0, 4, 5, 6, 7, 8, 43, 44,
#                45, 46, 47, 48, 49, 50]
# group_order_new = [0, 23, 24, 35, 12, 10]
group_order_new = [38, 23, 34, 40, 39, 37, 36, 35, 12, 10, 25, 26, 27]
# group_order_new = range(51)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """

    data = loadmat(os.path.join(save_dir, 'RAP_annotation/RAP_annotation.mat'))

    dataset = EasyDict()
    dataset.description = 'rap'
    dataset.reorder = 'group_order'
    dataset.root = os.path.join(save_dir, 'RAP_dataset')
    dataset.image_name = [dataset.root + '/' + data['RAP_annotation'][0][0][5][i][0][0] for i in range(41585)]
    raw_attr_name = [data['RAP_annotation'][0][0][3][i][0][0] for i in range(92)]
    # (41585, 92)
    raw_label = data['RAP_annotation'][0][0][1]
    new_label = np.zeros((41585, 13)).tolist()
    for step, label in enumerate(raw_label):
        for i in range(13):
            new_label[step][i] = label[group_order_new[i]]
        if new_label[step][10] == 1 or new_label[step][11] == 1 or new_label[step][12] == 1:
            new_label[step][10] = 1
        if new_label[step][3] == 1 or new_label[step][4] == 1 or new_label[step][5] == 1 or new_label[step][6] == 1 or new_label[step][7] == 1:
            new_label[step][3] = 1
        new_label[step].pop(-1)
        new_label[step].pop(-1)
        for i in range(4):
            new_label[step].pop(4)
    dataset.label = new_label
    count = [0]*7
    for i in new_label:
        for j in range(len(i)):
            count[j] += i[j]
    for i in count:
        print(i)
    print(len(new_label))
    print()
    # count_W_S = 0
    # count_W_L = 0
    # count_M_S = 0
    # count_M_L = 0
    # for L in new_label:
    #     if L[0] == 1 and L[-2] == 1:
    #         count_W_L += 1
    #     if L[0] == 1 and L[-2] == 0:
    #         count_W_S += 1
    #     if L[0] == 0 and L[-2] == 1:
    #         count_M_L += 1
    #     if L[0] == 0 and L[-2] == 0:
    #         count_M_S += 1
    # print(count_W_S, count_W_L, count_M_S, count_M_L)
    dataset.attr_name = [raw_attr_name[i] for i in group_order_new]

    # if reorder:
    #     dataset.label = dataset.label[:, np.array(group_order)]
    #     dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.trainval = []
    dataset.partition.test = []

    dataset.weight_trainval = []

    for idx in range(5):
        trainval = data['RAP_annotation'][0][0][0][idx][0][0][0][0][0, :] - 1
        test = data['RAP_annotation'][0][0][0][idx][0][0][0][1][0, :] - 1

        dataset.partition.trainval.append(trainval)
        dataset.partition.test.append(test)

        # weight_trainval = np.mean(dataset.label[trainval], axis=0).astype(np.float32)
        # dataset.weight_trainval.append(weight_trainval)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    save_dir = '/data1/zhengxiaoqiang/par_datasets/RAP/'
    reorder = True
    generate_data_description(save_dir, reorder)
