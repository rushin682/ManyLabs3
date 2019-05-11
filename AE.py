#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pandas as pd
import numpy as np
from os.path import exists, isdir, isfile, join
import random


import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)


from dataloader import ML3Dataset, get_category_level_data
from autoencoder import Autoencoder


# In[2]:


print("torch : {}".format(torch.__version__))
print("numpy : {}".format(np.__version__))
print("pandas : {}".format(pd.__version__))


# In[3]:


parser = argparse.ArgumentParser(description='Hyperparameters')
parser.add_argument('--data_dir', default='files', metavar='DIR', help='path to csv')
parser.add_argument('--category', default='intrinsic', metavar='CSV_File', help='The csv category')
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--epochs', default=13, type=int)
parser.add_argument('-b', '--batch-size', default=32, type=int, help='mini-batch size') #could use batch_size 12
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--weight-decay', default=0.1, type=float, help='weight decay')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
best_val_loss = 0

args = parser.parse_args()

device = torch.device("cpu")

data_dir = '/Users/bravos/Documents/CourseWork/Semester2/cs536-ML/ManyLab3/'
files_dir = join(data_dir, args.data_dir)
#all_data_dir = join(data_dir, 'ML3')

assert isdir(data_dir) and isdir(files_dir)

study_data = get_category_level_data(files_dir, category_type=args.category)
print(len(study_data['train']), len(study_data['valid']))

kwargs = {}

train_loader = data.DataLoader(
    ML3Dataset(study_data['train'], transform=None),
    batch_size=args.batch_size, shuffle=True, **kwargs)

valid_loader = data.DataLoader(
    ML3Dataset(study_data['valid'], transform=None),
    batch_size=args.batch_size, shuffle=False, **kwargs)


# In[ ]:


class VAE(nn.Module):
    
    def __init__(self, input_shape, n_latent):
        super(VAE, self).__init__()
        
        self.input_shape = input_shape
        self.n_latent = n_latent

        #encode
        self.fc1 = nn.Linear(self.input_shape, 8) #first dense connection
        self.fc2 = nn.Linear(8, 8) #Second dense connection -> 8 to 8
        self.fc3 = nn.Linear(8, self.n_latent) #Third Dense Connection -> 8 to latent
        #decode
        self.fc4 = nn.Linear(self.n_latent, 8) #4th Dense connection -> Latent to *
        self.fc5 = nn.Linear(8,8) #5th Dense connection -> 8 to 8
        self.fc6 = nn.Linear(8, self.input_shape) #Final Output Connection

    def encode(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        return h3

    def decode(self, h3):
        h4 = torch.tanh(self.fc4(h3))
        h5 = torch.tanh(self.fc5(h4))
        
        return h5
    
    def forward(self, x):
        h3 = self.encode(x)
        h5 = self.decode(h3)
        return self.fc6(h5)


# In[ ]:


#dataset_sizes = {x: len(study_datasets[x]) for x in ['train', 'valid']}          

model = VAE(input_shape=15, n_latent=4).to(device)

# We use an initial learning rate of 0.0001 that is decayed by a factor of
# 10 each time the validation loss plateaus after an epoch, and pick the
# model with the lowest validation loss
optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

loss_function = nn.L1Loss(reduction='sum')

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (row, _) in enumerate(train_loader):
        row = row.to(device)
        input = Variable(row)
        target = input.clone()
    
        optimizer.zero_grad()
        recon_batch = model(input)
        target.require_grad = False
        loss = loss_function(recon_batch, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(row), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(row)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def valid(epoch):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i, (row, _) in enumerate(valid_loader):
            row = row.to(device)
            input = Variable(row)
            target = input.clone()
            
            recon_batch = model(input)
            target.require_grad = False
            valid_loss += loss_function(recon_batch, target).item()
            if i == 0:
                n = min(row.size(0), 8)
#                comparison = torch.cat([row[:n],recon_batch[:n]])
                print (row[:n])
                print ("\n")
                print(recon_batch[:n])
#                 See what happens when you try to concatenate two torch rows. It should probably we one row elongated.
#                 But we need two rows to compare each index or two columns as such for the same.
    valid_loss /= len(valid_loader.dataset)
    print('====> Valid set loss: {:.4f}'.format(valid_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        valid(epoch)
#         with torch.no_grad():
#             sample = torch.randn(64, 20).to(device)
#             sample = model.decode(sample).cpu()
#             save_image(sample.view(64, 1, 28, 28),
#                        'results/sample_' + str(epoch) + '.png')


# In[ ]:




