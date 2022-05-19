# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:58:07 2020

@author: 86186
"""

import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import h5py
import utils
from model import Model#, Model_Attention


# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        #print(pos_1.shape, pos_2.shape, target.shape) # torch.Size([128, 3, 28, 28]) torch.Size([128, 3, 28, 28]) torch.Size([128])
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Multiview Learning for HSI classification')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=128, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=50, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs
    
    
    f=h5py.File('data/IP28-28-6.h5','r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()

    print(label, label.max(), label.min(), label.shape)
    
    # data prepare
    train_data = utils.HsiDataset(data, label, transform=utils.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    #model = Model_Attention(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 28, 28).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = train_data.classes

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0

    ############################################################################################
    # for epoch in range(1, epochs + 1):
    #     train_loss = train(model, train_loader, optimizer)
    # torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
    ############################################################################################

    model.load_state_dict(torch.load('results/128_0.5_200_128_50_model_ip_ip.pth'))

    f = h5py.File('data/IP28-28-6.h5', 'r')
    data = f['data'][:]
    label = f['label'][:]
    f.close()

    memory_data = utils.HsiDataset_test(data, label, transform=utils.test_transform)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    feature_bank=[]
    with torch.no_grad():
        for data, _, target in(memory_loader):
            #print(data.shape, target.shape) # torch.Size([128, 3, 28, 28]) torch.Size([128])
            feature, out = model(data.cuda(non_blocking=True))
            feature_bank.append(out)
    #print()
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    feature_labels = torch.tensor(memory_loader.dataset.label, device=feature_bank.device)
    x=feature_bank.cpu().numpy()
    y=feature_labels.cpu().numpy()
    x=x.T

    print(x.shape)
    print(y.shape)

    f=h5py.File('data/IP_feature.h5','w')
    f['data']=x
    f['label']=y
    f.close()