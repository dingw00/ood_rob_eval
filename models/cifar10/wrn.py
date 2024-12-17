
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, head_size, widen_factor=1, dropRate=0.0, use_norm=False, feature_norm=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6 # 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        if use_norm:
            self.fc = NormedLinear(nChannels[3], head_size)
        else:
            self.fc = nn.Linear(nChannels[3], head_size)

        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.feature_norm = feature_norm
        self.layer_list = ['conv1', 'block1', 'block2', 'block3', 'avg_pool', 'fc']

    def forward(self, x, return_feature=False):
        out = self.conv1(x) # (N, 16, 32, 32)
        out = self.block1(out) # (N, 32, 32, 32)
        out = self.block2(out) # (N, 64, 16, 16)
        out = self.block3(out) # (N, 128, 8, 8)
        out = self.relu(self.bn1(out)) # (N, 128, 8, 8)
        # here
        out = F.avg_pool2d(out, 8) # (N, 128, 1, 1)
        out = out.view(-1, self.nChannels) # (N, 128)
        if self.feature_norm:
            out = F.normalize(out, dim=1) * 25
        if return_feature:
            return self.fc(out), out
        return self.fc(out)
    
    def forward_features(self, x, layer="avg_pool"):
        assert layer in self.layer_list

        out = self.conv1(x)
        if layer == self.layer_list[0]:
            return out
        out = self.block1(out)
        if layer == self.layer_list[1]:
            return out
        out = self.block2(out)
        if layer == self.layer_list[2]:
            return out
        out = self.block3(out)
        if layer == self.layer_list[3]:
            return out
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.feature_norm:
            out = F.normalize(out, dim=1) * 25
        if layer == self.layer_list[4]:
            return out
        out = self.fc(out)
        if layer == self.layer_list[5]:
            return out

    def forward_head(self, x, layer):
        assert layer in self.layer_list
        
        if layer in self.layer_list[:1]:
            x = self.block1(x)
        if layer in self.layer_list[:2]:
            x = self.block2(x)
        if layer in self.layer_list[:3]:
            x = self.block3(x)
        if layer in self.layer_list[:4]:
            x = self.relu(self.bn1(x))
            x = F.avg_pool2d(x, 8)
            x = x.view(-1, self.nChannels)
            if self.feature_norm:
                x = F.normalize(x, dim=1) * 25
        if layer in self.layer_list[:5]:
            x = self.fc(x)
        return x

    def forward_feature_list(self, x):
        layer_list = ['conv1', 'block1', 'block2', 'block3', 'avg_pool', 'fc']
        out_list = []
        out = self.conv1(x) # layer minus_6
        out_list.append(out) 
        out = self.block1(out) # layer minus_5
        out_list.append(out)
        out = self.block2(out) # layer minus_4
        out_list.append(out)
        out = self.block3(out) # layer minus_3
        out_list.append(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8) 
        out = out.view(-1, self.nChannels)  # layer minus_2
        out_list.append(out)
        out = self.fc(out) # layer minus_1
        out_list.append(out)
        return layer_list, out_list


KNOWN_MODELS = OrderedDict([
    ('WRN-40-2', lambda *a, **kw: WideResNet(depth=40, widen_factor=2, *a, **kw)),
])
