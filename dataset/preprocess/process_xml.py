import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
class_label = ["gender", "hairlength", "hatclass", "TShirt", "Shirt", "Jacket", "Cotton", "Suit-Up",
               "LongTrousers", "Shorts", "Skirt", "ShortSkirt", "LeatherShoes", "SportShoes", "Sandal", "Boots",
               "SSBag", "Backpack", "Luggage"
               ]
# config_path = 'config_together.json'
# with open(config_path) as config_buffer:
#     config = json.loads(config_buffer.read())


def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs_list = []
    seen_labels = {}
    for ann in tqdm(sorted(os.listdir(ann_dir))):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                cls = {}
                for attr in list(elem):
                    print(attr)
                    if 'name' in attr.tag:
                        if attr.text in class_label:
                            cls['name'] = attr.text
                            if cls['name'] in seen_labels:
                                seen_labels[cls['name']] += 1
                            else:
                                seen_labels[cls['name']] = 1
                        else:
                            obj['name'] = attr.text
                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1
                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]

                    if 'class' in attr.tag:
                        for dim in list(attr):
                            img['%s' % cls['name']] = int(round(float(dim.text)))
        # train_src = 'train_class.txt'  # training annotations
        # val_src = 'val_class.txt'  # testing annotations
        if len(img['object']) > 0:
            all_imgs_list += [img]

    return all_imgs_list, seen_labels
ann_dir = '/data1/zhengxiaoqiang/par_datasets/new_dataset/IMAGES_LABEL/'
img_dir = '/data1/zhengxiaoqiang/par_datasets/new_dataset/IMAGES_TRAIN/'
all_imgs_list, seen_labels = parse_annotation(ann_dir, img_dir)
print(all_imgs_list[0])

