import torch
from torch import nn,optim
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def ConvBl(n_in, n_out):
    return nn.Sequential(nn.Conv2d(n_in, n_out, 3, 1, 1, bias= False),
                         nn.BatchNorm2d(n_out),
                         nn.ReLU())

class ResidualBlock(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(ConvBl(n_in,n_in),
                                 ConvBl(n_in, n_in))
    def forward(self,x):
        return x + self.net(x)

class ResNet9(nn.Module):
    def __init__(self, n_in, n_out): # n_out = number of class
        super().__init__()
        self.net = nn.Sequential(
            ConvBl(n_in, 64),
            ConvBl(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128),
            ConvBl(128, 256),
            nn.MaxPool2d(2),
            ConvBl(256, 512),
            nn.MaxPool2d(2),
            ResidualBlock(512),
            nn.MaxPool2d(4),
            nn.Flatten(),
            # nn.Linear(512, n_out)
        )
    def forward(self, images):
        output = self.net(images)
        return output
x = torch.randn(64,3,32,32)
model = ResNet9(3,10)
y = model(x)
print(y.shape)