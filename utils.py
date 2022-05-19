# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:49:51 2020

@author: 86186
"""

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image 

class HsiDataset(Dataset):
    def __init__(self, data, label, transform):
        self.data = data.reshape(-1,28,28,6)
        self.label = label
        self.transform = transform
        self.classes = label.max()+1

    def __getitem__(self,i):
        img1 = self.data[i,:,:,:3]
        img1 = Image.fromarray(img1)
        img1 = self.transform(img1)
        img2 = self.data[i,:,:,3:]
        img2 = Image.fromarray(img2)
        img2 = self.transform(img2)
        
        return img1, img2, self.label[i]

    def __len__(self):
        return len(self.data)


class HsiDataset_test(Dataset):
    def __init__(self, data, label, transform):
        self.data = data.reshape(-1, 28, 28, 6)
        self.label = label
        self.transform = transform
        self.classes = label.max() + 1

    def __getitem__(self, i):
        img1 = self.data[i, :, :, :3]
        img1 = Image.fromarray(img1)
        img1 = self.transform(img1)
        img2 = self.data[i, :, :, 3:]
        img2 = Image.fromarray(img2)
        img2 = self.transform(img2)

        return img1, img2, self.label[i]

    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28),#
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(), # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(), # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])