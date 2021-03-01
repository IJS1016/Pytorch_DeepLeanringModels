import os
import numpy as np
from datetime import datetime 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

# IMPORT LAYERS
from VGG import *
from Lenet5 import LeNet5
from Resnet import *
from dataloader import *

from torchvision.datasets import ImageFolder


#######################################################################
# Setting for options
#######################################################################
DEVICE  = torch.device('cuda:'+str(1) if torch.cuda.is_available() else 'cpu')
MODEL   = Resnet34
CLASS_N = 10
DATASET = datasets.CIFAR10

print(f"\n\nTRAINING STRAT with {MODEL}\n\n")

# Hyperparameter
learning_rate = 1e-2
batch_size = 128
num_epoch = 10

ckpt_path = "../checkpoint"
result_path = "../result"

#######################################################################
# Functions
#######################################################################
# to save and load
def save(ckpt_path, net, optim, epoch):
    if os.path.exists(ckpt_path) == False:
        os.makedirs(ckpt_path)
    torch.save({'net':net.state_dict(), 'optim': optim.state_dict()},'%s/net_epoch%d.pth' %(ckpt_path, epoch))

def load(ckpt_path, net, optim):
    if os.path.exists(ckpt_path) == False:
        os.mkdir(ckpt_path)
    ckpt_list = os.listdir(ckpt_path)
    ckpt_list.sort()
    model_dict = torch.load('%s/%s' % ckpt_path, ckpt_list[-1])
    net.load_state_dict(model_dict['net'])
    optim.load_state_dict(model_dict['optim'])
    
    return net, optim

# for init
import torch.nn.init as init
def init_weight(net, init_type='kaiming', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)


# define transforms
transforms = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# download and create datasets
datapath = '/home/ijs1016/ImageNet-100'
trainset = ImageFolder(root=datapath+'/train', transform=transforms)
testset = ImageFolder(root=datapath+'/test', transform=transforms)



# custom ImageDataset (!! HAVE TO FIXED) 
# trainset = ImageDataset(data_path=datapath, transforms_=None, mode = 'train')

# train_dataset = DATASET(root='../data_sets', 
#                         train=True, 
#                         transform=transforms,
#                         download=True)

# valid_dataset = DATASET(root='../data_sets', 
#                         train=False, 
#                         transform=transforms)



# define the data loaders
train_loader = DataLoader(dataset=trainset, 
                          batch_size=batch_size, 
                          shuffle=True)

valid_loader = DataLoader(dataset=testset, 
                          batch_size=batch_size, 
                          shuffle=False)
num_data = len(train_loader.dataset)
num_iter = np.ceil(num_data/batch_size)


#######################################################################
# TRAIN AND VALIDATE
#######################################################################
# network init set
net = MODEL(CLASS_N).to(DEVICE)
init_weight(net=net, init_type='kaiming', init_gain=0.02)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# training
for epoch in range(1, num_epoch+1):
    net.train()
    loss_list = []
    acc_list = []
    for _iter, (data, label) in enumerate(train_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        output, prob = net(data)
        _, predict = torch.max(prob, 1)

        optimizer.zero_grad()

        loss = criterion(output, label)
        acc = ((predict == label).type(torch.float)).mean()

        loss.backward()
        optimizer.step()

        loss_list += [loss.item()]
        acc_list += [acc.item()]

        if _iter %10 ==0 :
            print('Train: Epoch %04d/%04d | Iteration %04d/%04d | Loss: %.4f | Acc %.4f'%
                (epoch, num_epoch, _iter, num_iter, np.mean(loss_list), np.mean(acc_list)))

    save(ckpt_path, net, optimizer, epoch)

# validation
with torch.no_grad():
    net.eval()
    loss_list = []
    acc_list = []
    for batch, (data, label) in enumerate(valid_loader):
        data = data.to(DEVICE)
        label = label.to(DEVICE)

        output, prob = net(data)
        predict = torch.max(prob, 1)

        loss = criterion(output, label)
        acc = ((predict[1] == label).type(torch.float)).mean()
        loss_list += [loss.item()]
        acc_list += [acc.item()]

        print('Test: Loss: %.4f | Acc %.4f'%
              ( np.mean(loss_list), np.mean(acc_list)))





# ???????????
# DO NOT USED
# ???????????
def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} — '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)