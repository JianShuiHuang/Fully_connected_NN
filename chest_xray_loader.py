# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:25:44 2020

@author: ivis
"""
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder


def dataLoader(mode):  
    if mode == 'train':
        train_dataset = ImageFolder(root="chest_xray/train/")
        
        print("> Found %d images..." % (len(train_dataset)))
        
        train_data = np.zeros((len(train_dataset), 3 * 64 * 64))
        train_label = np.zeros(len(train_dataset))
        
        for i in range(len(train_dataset)):
            img = img = Image.open(train_dataset.imgs[i][0])
            img = img.resize((64, 64),Image.ANTIALIAS)
            img = img.convert('RGB')
            
            img_np = np.asarray(img)/255
            img_np = np.transpose(img_np, (2,0,1))

            train_data[i] = img_np.flatten()
            train_label[i] = train_dataset.imgs[i][1]         
            
        state = np.random.get_state()
        np.random.shuffle(train_data)
        np.random.set_state(state)
        np.random.shuffle(train_label)
            
        return train_data, train_label
    
    else:
        test_dataset = ImageFolder(root="chest_xray/test/")
        
        print("> Found %d images..." % (len(test_dataset)))
        
        test_data = np.zeros((len(test_dataset), 3 * 64 * 64))
        test_label = np.zeros(len(test_dataset))
        
        for i in range(len(test_dataset)):
            img = img = Image.open(test_dataset.imgs[i][0])
            img = img.resize((64, 64),Image.ANTIALIAS)
            img = img.convert('RGB')
            
            img_np = np.asarray(img)/255
            img_np = np.transpose(img_np, (2,0,1))

            test_data[i] = img_np.flatten()
            test_label[i] = test_dataset.imgs[i][1]         
            
            
        return test_data, test_label
        


