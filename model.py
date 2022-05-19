# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:52:59 2020

@author: 86186
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

from torchvision.models import resnext50_32x4d
from torchvision.models import wide_resnet50_2
from torchvision.models import densenet161
#from cbam_resnext import cbam_resnext50_16x64d


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
        #for name, module in resnext50_32x4d().named_children():
        #for name, module in wide_resnet50_2().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))#2048

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
    
# class Model_Attention(nn.Module):
#     def __init__(self, feature_dim=128):
#         super(Model_Attention, self).__init__()

#         self.f = []
#         self.f = cbam_resnext50_16x64d(128)
#         self.g = nn.Sequential(nn.Linear(128, 512, bias=False), nn.BatchNorm1d(512),
#                                nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

#     def forward(self, x):
#         out1 = self.f(x)
#         out2 = self.g(out1)
#         return F.normalize(out1, dim=-1), F.normalize(out2, dim=-1)
    
    
#class Model1(nn.Module):
#    def __init__(self, feature_dim=128):
#        super(Model1, self).__init__()
#
#        self.f = []
#        for name, module in resnet50().named_children():
#            if name == 'conv1':
#                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                self.f.append(module)
#        # encoder
#        self.f = nn.Sequential(*self.f)
#
#    def forward(self, x):
#        x = self.f(x)
#        feature = torch.flatten(x, start_dim=1)
#        return feature