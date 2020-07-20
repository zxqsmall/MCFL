import math
import torch
import torch.nn as nn
import numpy as np
__all__ = ['CBAMLayer', 'SPPLayer']


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        # channel attention
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmod(max_out + avg_out)
        x = channel_out * x

        # spatial attention
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmod(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class SPPLayer(nn.Module):
    def __init__(self, pool_size, pool=nn.MaxPool2d):
        super(SPPLayer, self).__init__()
        self.pool_size = pool_size
        self.pool = pool
        self.out_length = np.sum(np.array(self.pool_size) ** 2)

    def forward(self, x):
        B, C, H, W = x.size()
        for i in range(len(self.pool_size)):
            h_wid = int(math.ceil(H / self.pool_size[i]))
            w_wid = int(math.ceil(W / self.pool_size[i]))
            h_pad = (h_wid * self.pool_size[i] - H + 1) / 2
            w_pad = (w_wid * self.pool_size[i] - W + 1) / 2
            out = self.pool((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))(x)
            if i == 0:
                spp = out.view(B, -1)
            else:
                spp = torch.cat([spp, out.view(B, -1)], dim=1)
        return spp

