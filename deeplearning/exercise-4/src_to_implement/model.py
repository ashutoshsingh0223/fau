import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch import flatten

class ResBlock(nn.Module):
    def __init__(self ,in_channels, out_channels , stride):
        super(ResBlock,self).__init__()
        # nn.Conv2d(in_channels, out_channels, filter_size=3, stride)
        self.out_channels = out_channels
        self.stride = stride
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=[1, 1])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.cn2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=[1, 1])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride)
        self.residual_norm = nn.BatchNorm2d(out_channels)

        # self.downsample = downsample

    def forward(self, x):
        res_cn = self.residual_conv(x)
        x = self.cn1(x)
        res_norm = self.residual_norm(res_cn)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.cn2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x + res_norm

    def __call__(self, x):
        return self.forward(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)
        self.l1 = ResBlock(64, 64, 1) #resnet 64
        self.l2 = ResBlock(64, 128, 2) #resnet 128
        self.l3 = ResBlock(128, 256, 2) #resnet 256
        self.l4 = ResBlock(256, 512, 2) #resnet 512
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.cv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.mp(out)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.global_avg_pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        # out = self.sigmoid(out)
        return out

