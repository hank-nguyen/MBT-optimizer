# -*- coding: utf-8 -*-
"""
Accompanying implementation for paper:
BACKTRACKING GRADIENT DESCENT METHOD FOR GENERAL C1 FUNCTIONS, WITH APPLICATIONS TO DEEP LEARNING
https://arxiv.org/pdf/1808.05160.pdf

Forked and inspired by https://github.com/kuangliu/pytorch-cifar
Train CIFAR10 with PyTorch
Learning rate finder using Backtracking line search with different batch sizes
and different starting learning rates
"""
#from log import backup, login, logout; backup(); login()
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
import json, pickle

from models import *
from utils import count_parameters, dataset
from lr_backtrack import LRFinder, change_lr

all_batch_sizes = [12,25,50,100,200,400,800]
all_lr_starts   = [100,10,1,0.1,0.01,0.001,1e-4,1e-5,1e-6]

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_loss = loss_avg = 1e10 # best (smallest) training loss 
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
patient = 0 # number of epochs waiting for improvement of best_acc or best_loss

cifar_dataset = 10# CIFAR100 or 100
num_classes = cifar_dataset
momentum = 0.9

# Backtracking hyper-parameters
BT = 1 # using backtracking or not
lr_justified = True
alpha = 1e-4
beta = 0.5
num_iter = 20

save_paths = ['weights/','history','history/lr']
for save_path in save_paths:
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
# Model
net = ResNet18(num_classes);
net_name = 'ResNet18'

print('Model:',net_name)
print('Number of parameters:',count_parameters(net),'Numbers of Layers:', len(list(net.parameters())))
net = net.to(device)

save_dir = save_path + net_name + '/'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

weights_init = save_dir + net_name+'_CF'+str(num_classes)+'_init.t7'
weights_best = save_dir + net_name+'_CF'+str(num_classes)+'_best.t7'
history_path = save_dir + net_name+'_CF'+str(num_classes)+'_history.json'

#cuda device
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('Resuming from checkpoint %s..'%weights_best)
    assert os.path.isfile(weights_best), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(weights_best)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
all_history = {}

lr_full = {}
# loop for batch size
for batch_size in all_batch_sizes:
    lr_full[batch_size] = {}
    trainloader, testloader, num_batches = dataset(cifar_dataset, batch_size)
    # loop for starting learning rate
    for lr_start in [100,10,1,0.1,0.01,0.001,1e-4,1e-5,1e-6]:
        optimizer_BT = optim.SGD(net.parameters(), lr=lr_start)
        print('Start learning rate:',optimizer_BT.param_groups[0]['lr'])
        lr_finder_BT = LRFinder(net, optimizer_BT, criterion, device="cuda")
        print("Using backtrack with", optimizer_BT.__class__.__name__, ", alpha =",alpha, ', beta =',beta)
        lr_finder_BT.backtrack(trainloader, alpha = alpha, beta = beta, num_iter = num_iter, lr_justified = lr_justified)
        lr_full[batch_size][lr_start] = lr_finder_BT.lr_BT
        
        json.dump(lr_full, open("history/lr/lr_full_cifar%d.json"%num_classes, 'w'),indent = 4)
        pickle.dump(lr_full,open('history/lr/lr_full_cifar%d.pickle'%num_classes,'wb'))
        
# print result
print('Learning rate finding result using Backtracking line search with different batch sizes:')
print(lr_full)

