# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 09:30:21 2020

@author: ivis
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models 
import pandas as pd
from torch.utils import data
from PIL import Image
from torchvision import transforms
from NN import *
from DataLoader import *
from Time import *
import time
from TrainTest import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Hyper parameters
lr = 1e-03
BatchSize = 4
Epochs = 10
Momentum = 0.9     
Weight_decay = 5e-4
in_feature = 3 * 64 * 64
out_feature = 1000
"""
transformations = transforms.Compose([transforms.ToTensor()])
train_np = DataLoader('data/', 'train', transformations)
test_np = DataLoader('data/', 'test', transformations)
    
num_images_train = 28099
num_images_test = 7025
input_size = train_np[0][0].shape[0] * train_np[0][0].shape[1] * train_np[0][0].shape[2]
output_size = 5
hidden_size = 1000    
    
print(input_size)
 
train_data = np.zeros((num_images_train, input_size))
train_label = np.zeros(num_images_train)
for i in range(num_images_train):
    if i % 100 == 0:
        print(i)
    train_data[i] = train_np[i][0].flatten()
    train_label[i] = 1 if train_np[i][1] >= 1 else 0

np.save("Retinopathy_train_data.npy", train_data)
np.save("Retinopathy_train_label.npy", train_label)


test_data = np.zeros((num_images_test, input_size))
test_label = np.zeros(num_images_test)
for i in range(num_images_test):
    if i % 100 == 0:
        print(i)
    test_data[i] = test_np[i][0].flatten()
    test_label[i] = 1 if test_np[i][1] >= 1 else 0

np.save("Retinopathy_test_data.npy", test_data)
np.save("Retinopathy_test_label.npy", test_label)
"""
    

train_data = np.load("Retinopathy_train_data.npy")
train_label = np.load("Retinopathy_train_label.npy")
test_data = np.load("Retinopathy_test_data.npy")
test_label = np.load("Retinopathy_test_label.npy")

print("data load complet...")

print(train_data.shape)
print(train_label.shape)

print(train_label)

"""
train_accuracy = []
test_accuracy = []
    
model = fully_connected(in_feature, out_feature)
#model = torch.load('preresnet18.pkl').to(device)
    
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = Momentum, weight_decay = Weight_decay)
    
start = time.time()

for i in range(Epochs):
    train = Train(train_data, train_label, model, optimizer, BatchSize)
    train_accuracy.append(train)
    test = Test(test_data, test_label, model, BatchSize)
    test_accuracy.append(test)
    print("epochs:", i )
    print('Train Accuracy: ', train)
    print('Test Accuracy: ', test)

print("Time: ", timeSince(start, 1 / 100))

print('Max accuracy: ', max(test_accuracy))
print("model complet...")
"""
    
    
    
    
