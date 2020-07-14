import os
from tqdm import tqdm
from easydict import EasyDict
import xml.etree.ElementTree as ET
import pickle
import random


def generate_data_description(img_dir, ann_dir, save_dir):
    dataset = EasyDict()
    img_name = []
    att_name = []
    labels = []
    selected_label = [0, 2, 7, 16, 17, 15, 1, 9, 10]
    for ann in tqdm(sorted(os.listdir(ann_dir))):
        tree = ET.parse(ann_dir + ann)
        label = []
        for elem in tree.iter():
            if 'filename' in elem.tag:
                image_name = img_dir + elem.text
                img_name.append(image_name)
            if 'object' in elem.tag or 'part' in elem.tag:
                for attr in list(elem):
                    if len(att_name) <= 18 and 'name' in attr.tag:

                        att_name.append(attr.text)
                    if 'class' in attr.tag:
                        for att in list(attr):
                            label.append(int(att.text))
        label = [label[i] for i in selected_label]
        print(label)
        label[6] = 1 - label[6]
        if label[7] == 1 or label[8] == 1:
            label[7] = 1
        if label[3] == 1 or label[4] == 1:
            label[3] = 1
        label.pop(4)
        label.pop(-1)
        # print(label)
        labels.append(label)
    dataset.image_name = img_name
    dataset.label = labels
    dataset.attr_name = att_name
    train_num = int(0.8 * len(img_name))
    img_idx = random.sample(range(20000), train_num)
    dataset.img_idx = img_idx
    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    ann_dir = '/data1/zhengxiaoqiang/par_datasets/new_dataset/IMAGES_LABEL/'
    img_dir = '/data1/zhengxiaoqiang/par_datasets/new_dataset/IMAGES_TRAIN/'
    save_dir = '/home/visitor2/baseline-MCFL'
    generate_data_description(img_dir, ann_dir, save_dir)