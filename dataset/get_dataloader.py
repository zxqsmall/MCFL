from dataset.AttrDataset import get_transform
from dataset.AttrDataset import AttrDataset, AttrDataset_for_newdataset
from torch.utils.data import DataLoader
import torch.utils.data as data


def get_dataloader():
    train_transform, valid_transform, _ = get_transform()
    data_path_rap = '/data1/zhengxiaoqiang/par_datasets/RAP/dataset.pkl'
    data_path_pa100k = '/data1/zhengxiaoqiang/par_datasets/PAK100/dataset.pkl'
    data_path_peta = '/data1/zhengxiaoqiang/par_datasets/PETA/peta-release/dataset.pkl'
    train_set = AttrDataset(split='trainval', transform=train_transform)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=100,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    valid_set = AttrDataset(split='test', transform=valid_transform)
    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    labels = train_set.label

    return train_dataloader, valid_dataloader, labels


def get_dataloader_new():
    train_transform, valid_transform, target_transform = get_transform()
    train_set = AttrDataset_for_newdataset(split='trainval', transform=train_transform, target_transform=target_transform)
    test_set = AttrDataset_for_newdataset(split='test', transform=train_transform)

    # train_set, valid_set = data.random_split(train_set, [int(0.8*len(train_set)), len(train_set) - int(0.8*len(train_set))])
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=200,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    # valid_set = AttrDataset(split='test', transform=valid_transform)
    valid_dataloader = DataLoader(
        dataset=test_set,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    labels = train_set.label

    return train_dataloader, valid_dataloader, labels