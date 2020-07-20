import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from model.backbones.resnet import resnet50
from config.defaults_config import cfg


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.num_att = cfg.model.num_att
        self.last_conv_stride = cfg.model.last_conv_stride
        self.drop_pool5 = cfg.model.drop_pool5
        self.drop_pool5_rate = cfg.model.drop_pool5_rate
        self.pretrained = cfg.model.pretrained
        self.base50 = resnet50(pretrained=self.pretrained, last_conv_stride=self.last_conv_stride)
        self.batch_norm = nn.BatchNorm1d(self.num_att)
        self.classifier = nn.Linear(2048, self.num_att)
        init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = self.base50(x)
        x = F.max_pool2d(x, x.shape[2:])
        x = x.view(x.size(0), -1)
        if self.drop_pool5:
            x = F.dropout(x, p=self.drop_pool5_rate, training=self.training)
        x = self.classifier(x)
        x = self.batch_norm(x)
        return x



