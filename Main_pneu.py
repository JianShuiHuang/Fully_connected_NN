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
from chest_xray_loader import *
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
out_feature = 15
"""
train_data, train_label = dataLoader('train')
test_data, test_label = dataLoader('test')

np.save("Pneunomia_train_data.npy", train_data)
np.save("Pneunomia_train_label.npy", train_label)
np.save("Pneunomia_test_data.npy", test_data)
np.save("Pneunomia_test_label.npy", test_label)
"""

train_data = np.load("Pneunomia_train_data.npy")
train_label = np.load("Pneunomia_train_label.npy")
test_data = np.load("Pneunomia_test_data.npy")
test_label = np.load("Pneunomia_test_label.npy")


print("data load complet...")
 
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
