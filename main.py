import torch
from torch import nn,optim
import numpy as np
import pandas as pd
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from Resnet import ResNet9
from Inceptionnet import InceptionNet
from torchvision.models import googlenet

device = torch.device('cuda')
epochs = 30
batch_size = 64
img_size = 32
img_channel = 3
n_class = 10

traindata = CIFAR10(root= './cifar10', 
                    train= True, 
                    download= True, 
                    transform= transforms.Compose([# transforms.Resize(img_size),
                                                   transforms.RandomCrop(img_size, padding = 4, padding_mode='reflect'),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5]*3, [0.5]*3)]))

testdata = CIFAR10(root= './CNN/cifar10', 
                    train= False, 
                    download= True, 
                    transform= transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize([0.5]*3, [0.5]*3)]))

train_loader = DataLoader(dataset= traindata, batch_size= batch_size, shuffle= True)
test_loader = DataLoader(dataset= testdata, batch_size= 128, shuffle= False)

model1 = ResNet9(3,10).to(device)
model2 = InceptionNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)

for epoch in range(10): 
    model2.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")


model2.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model2(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

# ResNet9 Accuracy: 88.56%