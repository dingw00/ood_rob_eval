
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torchvision.models.resnet import ResNet as ResNet_Imagenet_


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks=[3,4,6,3], num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.layer_list = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avg_pool', 'linear']

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # (N, 64, 32, 32)
        out = self.layer1(out) # (N, 64*4, 32, 32)
        out = self.layer2(out) # (N, 128*4, 16, 16)
        out = self.layer3(out) # (N, 256*4, 8, 8)
        out = self.layer4(out) # (N, 512*4 , 4, 4)
        out = F.avg_pool2d(out, 4) # (N, 512*4, 1, 1)
        out = out.view(out.size(0), -1)  # (N, 512*4)
        # features = out
        out = self.linear(out) # (N, num_classes)
        return out
    
    def forward_features(self, x, layer="avg_pool"):
        assert layer in self.layer_list

        out = F.relu(self.bn1(self.conv1(x)))
        if layer == 'conv1':
            return out
        out = self.layer1(out)
        if layer == 'layer1':
            return out
        out = self.layer2(out)
        if layer == 'layer2':
            return out
        out = self.layer3(out)
        if layer == 'layer3':
            return out
        out = self.layer4(out)
        if layer == 'layer4':
            return out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if layer == 'avg_pool':
            return out
        out = self.linear(out)
        if layer == 'linear':
            return out

    def forward_head(self, x, layer):
        assert layer in self.layer_list
        
        if layer in self.layer_list[:1]:
            x = self.layer1(x)
        if layer in self.layer_list[:2]:
            x = self.layer2(x)
        if layer in self.layer_list[:3]:
            x = self.layer3(x)
        if layer in self.layer_list[:4]:
            x = self.layer4(x)
        if layer in self.layer_list[:5]:
            x = F.avg_pool2d(x, 4)
            x = x.view(x.size(0), -1)
        if layer in self.layer_list[:6]:
            x = self.linear(x)
        return x

class ResNet_Imagenet(ResNet_Imagenet_):

    def __init__(self, block, num_blocks=[3,4,6,3], num_classes=100):
        super(ResNet_Imagenet, self).__init__(block, num_blocks, num_classes)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_list = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avg_pool', 'fc']

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) 
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # features = out
        out = self.fc(out) # (N, num_classes)
        return out
    
    def forward_features(self, x, layer="avg_pool"):
        assert layer in self.layer_list

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        if layer == 'conv1':
            return out
        out = self.layer1(out)
        if layer == 'layer1':
            return out
        out = self.layer2(out)
        if layer == 'layer2':
            return out
        out = self.layer3(out)
        if layer == 'layer3':
            return out
        out = self.layer4(out)
        if layer == 'layer4':
            return out
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        if layer == 'avg_pool':
            return out
        out = self.fc(out)
        if layer == 'fc':
            return out

    def forward_head(self, x, layer):
        assert layer in self.layer_list
        
        if layer in self.layer_list[:1]:
            x = self.layer1(x)
        if layer in self.layer_list[:2]:
            x = self.layer2(x)
        if layer in self.layer_list[:3]:
            x = self.layer3(x)
        if layer in self.layer_list[:4]:
            x = self.layer4(x)
        if layer in self.layer_list[:5]:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        if layer in self.layer_list[:6]:
            x = self.fc(x)
        return x
