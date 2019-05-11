#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

import re
import os
from os import getcwd
from os.path import join, exists, isdir, isfile
from tqdm import tqdm

import torch
import torch.utils.data as data


import numpy as np
import pandas as pd


# In[2]:


def get_category_level_data(data_dir, category_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as record dataframes,
    these dataframes contain multiple columns based on the category type
    Args:
        Category_type (string): one of the several study type file names in 'train/valid/test' dataset
    """
    
    study_data = {}
    #BASE_FILE = category_type + '.csv'
    
    os.chdir(data_dir)
    K = pd.read_csv('intrinsic.csv')
    K = K.dropna(axis = 0, how = 'all')
    K = K.replace(to_replace = float('NaN'), value = 0)
    
    msk = np.random.rand(len(K)) < 0.8
    study_data['train'] = K[msk]
    study_data['valid'] = K[~msk]
        
    return study_data


# In[3]:


class ML3Dataset(data.Dataset):

    def __init__(self, df, transform=None, download=False):
        
        self.df = df        
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        
        reconstructed_row = row.copy()
        
        return torch.tensor(row), torch.tensor(reconstructed_row)


# In[4]:


if __name__ == '__main__':
    import pprint

    data_dir = '/Users/bravos/Documents/CourseWork/Semester2/cs536-ML/ManyLab3/'
    files_dir = join(data_dir, 'files')
    #all_data_dir = join(data_dir, 'ML3')
    
    assert isdir(data_dir) and isdir(files_dir) #and isdir(all_data_dir)    
    
    study_data = get_category_level_data(files_dir, category_type='intrinsic')
    print(len(study_data['train']), len(study_data['valid']))
        
    #val_csv = join(data_dir, 'valid.csv')
    val_loader = data.DataLoader(
        ML3Dataset(study_data['valid']),
        batch_size=4,
        shuffle=False,
        num_workers=1,
        pin_memory=False)
        
    print("What is val_loader? :", type(val_loader))

    
    for i, (row, reconstructed_row) in enumerate(val_loader):
#         print(i, "\n")
        pprint.pprint(row)
        if i == 10:
            break


# In[ ]:




