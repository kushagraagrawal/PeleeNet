from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from collections import OrderedDict
import math

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation
    
    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.activation:
            out = F.relu(out,inplace=True)
        return out

class conv_relu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bottleneck_width, drop_rate):
        super(_DenseLayer, self).__init__()
        growth_rate = growth_rate // 2
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4
        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to', inter_channel)
        
        self.branch1a = conv_bn_relu(
            num_input_features, inter_channel, kernel_size=1)
        self.branch1b = conv_bn_relu(
            inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = conv_bn_relu(
            num_input_features, inter_channel, kernel_size=1)
        self.branch2b = conv_bn_relu(
            inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = conv_bn_relu(
            growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.branch1a(x)
        out1 = self.branch1b(out1)

        out2 = self.branch2a(x)
        out2 = self.branch2b(out2)
        out2 = self.branch2c(out2)

        out = torch.cat([x, out1, out2], dim=1)
        return out

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)

class _StemBlock(nn.Module):
    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features / 2)
        self.stem1 = conv_bn_relu(
            num_input_channels,num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem2a = conv_bn_relu(
            num_init_features, num_stem_features, kernel_size=1, stride=1, padding=0)
        self.stem2b = conv_bn_relu(
            num_stem_features, num_init_features, kernel_size=3, stride=2, padding=1)
        self.stem3 = conv_bn_relu(
            2*num_init_features, num_init_features, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self,x):
        out = self.stem1(x)
        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(out)
        return out
        
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.res1a = conv_relu(in_channels, 128, kernel_size=1)
        self.res1b = conv_relu(128, 128, kernel_size=3, padding=1)
        self.res1c = conv_relu(128, 256, kernel_size=1)
        self.res2a = conv_relu(in_channels, 256, kernel_size=1)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)
        out2 = self.res2a(x)
        out = out1 + out2
        return out