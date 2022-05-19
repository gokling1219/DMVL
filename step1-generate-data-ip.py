#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:14:15 2018

@author: liubing
"""

import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
import cv2

img=sio.loadmat('D:\hyperspectral_data\Indian_pines.mat')
img=img['indian_pines_corrected']

gt=sio.loadmat('D:\hyperspectral_data\Indian_pines_gt.mat')
gt=gt['indian_pines_gt']

m,n=gt.shape

num=np.zeros(17)
for i in range(m):
    for j in range(n):
       num[gt[i,j]]=num[gt[i,j]]+1 

def Patch(data,height_index,width_index,PATCH_SIZE):
    height_slice = slice(height_index-PATCH_SIZE, height_index+PATCH_SIZE)
    width_slice = slice(width_index-PATCH_SIZE, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice,:]
    patch = patch.reshape(-1,patch.shape[0]*patch.shape[1]*patch.shape[2])
    return patch
#xxx = img.max()
#img = img/xxx

img1 = img[:,:,:100]
img2 = img[:,:,100:]

img1 = img1.reshape(-1,100)
img2 = img2.reshape(-1,100)

pca=PCA(n_components=3)

reduced_img1 = pca.fit_transform(img1)
img_max, img_min = reduced_img1.max(), reduced_img1.min()
img1 = (reduced_img1-img_min)/(img_max-img_min)
img1 = img1.reshape(m,n,3)

reduced_img2 = pca.fit_transform(img2)
img_max, img_min = reduced_img2.max(), reduced_img2.min()
img2 = (reduced_img2-img_min)/(img_max-img_min)
img2 = img2.reshape(m,n,3)

PATCH_SIZE=14

img1=cv2.copyMakeBorder(img1, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
img2=cv2.copyMakeBorder(img2, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0,0,0))
gt=cv2.copyMakeBorder(gt, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, cv2.BORDER_CONSTANT, value=(0))


[mm,nn,bb]=img1.shape

data = []
label = []
gt_index = []

for i in range(PATCH_SIZE,mm-PATCH_SIZE):
    for j in range(PATCH_SIZE,nn-PATCH_SIZE):
        if gt[i,j]==0:
            continue
        else:
            temp_data1=Patch(img1,i,j,PATCH_SIZE)
            temp_data2=Patch(img2,i,j,PATCH_SIZE)
            temp_data1 = temp_data1.reshape(28,28,3)
            temp_data2 = temp_data2.reshape(28,28,3)
            temp_data = np.concatenate((temp_data1,temp_data2),2)
            #temp = np.swapaxes(temp_data,1,2)
            #temp_data = np.swapaxes(temp,0,1)
            temp_data = temp_data.reshape(-1)
            data.append(temp_data)
            label.append(gt[i,j]-1)
            gt_index.append((i-PATCH_SIZE)*n+j-PATCH_SIZE)
# gt_index=np.array(gt_index)
# np.save('ip_index',gt_index)

data=np.array(data)
data=np.squeeze(data)
data = np.uint8(data*255)
label=np.array(label)
label=np.squeeze(label)


import h5py
f=h5py.File('data/IP28-28-6.h5','w')
f['data']=data
f['label']=label
f.close()