import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch import flatten

class ResBlock(nn.Module):
    def __init__(self ,in_channel ,out_channel , stride = 1 , downsample = None):
        super(ResBlock,self).__init__()
        self.cv1= nn.Conv2d(in_channel , out_channel , stride )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.cv2 = nn.Conv2d(in_channel,out_channel,stride)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self , x):
         residual = x
         l= self.cv1(x)
         l = self.bn1(l)
         l = self.relu(l)
         l = self.cv2(l)
         l = self.bn2(l)
         if self.downsample:
             residual = self.downsample(x)
         l += residual
         l = self.relu(l)  #doubt here
         return l
'Resnet'
class ResNet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
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
        out = flatten(out, start_dim=1)
        out = self.fc(out)
        out =  F.sigmoid(out)
        return (out)



