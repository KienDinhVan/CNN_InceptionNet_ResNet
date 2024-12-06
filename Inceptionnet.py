import torch
from torch import nn,optim
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def ConvBl(n_in, n_out, k, s, p):
    return nn.Sequential(nn.Conv2d(n_in, n_out, k, s, p, bias= False),
                         nn.BatchNorm2d(n_out),
                         nn.ReLU())

class InceptionModule(nn.Module):
    def __init__(self, n_in, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_2):
        super().__init__()
        self.branch1 = ConvBl(n_in, out_1x1, 1, 1, 0)

        self.branch2 = nn.Sequential(
            ConvBl(n_in, red_3x3, 1, 1, 0),
            ConvBl(red_3x3, out_3x3, 3, 1, 1)
        )
        
        self.branch3 = nn.Sequential(
            ConvBl(n_in, red_5x5, 1, 1, 0),
            ConvBl(red_5x5, out_5x5, 5, 1, 2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBl(n_in, out_1x1_2, 1, 1, 0)
        )
    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim = 1)

# x = torch.randn(64,3,32,32)
# model = InceptionModule(3, 32, 32, 64, 16, 32, 32)
# y = model(x)
# print(y.shape)

class InceptionNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            ConvBl(3, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            InceptionModule(64, 32, 32, 64, 16, 32, 32),
            InceptionModule(160, 64, 64, 128, 32, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(320, 128, 128, 256, 64, 128, 128),
            InceptionModule(640, 128, 128, 256, 64, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(640, num_classes)
        )

    def forward(self, x):
        output = self.net(x)
        return output

x = torch.randn(64,3,32,32)
model = InceptionNet()
y = model(x)
print(y.shape)