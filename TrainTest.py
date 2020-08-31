import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import time
from Time import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Train(train_data, train_label, model, optimizer, BatchSize):
    ##training
    model.train() 
    true = 0
    false = 0
    Loss = nn.CrossEntropyLoss()   #change loss function here
        
    for j in range(len(train_data)//BatchSize + 1):
        l = j * BatchSize
        r = j * BatchSize + BatchSize
        if(r > len(train_data)):
            r = len(train_data)
        
        
        x_train = torch.from_numpy(train_data[l:r])
        y_train = torch.from_numpy(train_label[l:r])
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_train = Variable(x_train)
        y_train = Variable(y_train)
            
        prediction = model(x_train.float())
        
        for k in range(l, r):
            if (prediction[k-l][0] > prediction[k-l][1]) and (train_label[k] == 0):
                true = true + 1
            elif (prediction[k-l][0] > prediction[k-l][1]) and (train_label[k] == 1):
                false = false + 1
            elif (prediction[k-l][0] <= prediction[k-l][1]) and (train_label[k] == 1):
                true = true + 1
            else:
                false = false + 1
            
        loss = Loss(prediction, y_train.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return true / (true + false)

def Test( test_data, test_label, model, BatchSize):
    ##testing
    model.eval()
    true = 0
    false = 0
        
    for j in range(len(test_data)//BatchSize + 1):
        l = j * BatchSize
        r = j * BatchSize + BatchSize
        if(r > len(test_data)):
            r = len(test_data)
        
        x_test = torch.from_numpy(test_data[l:r])
        x_test = x_test.to(device)
        x_test = Variable(x_test)
        y_test = torch.from_numpy(test_data[l:r])
        y_test = x_test.to(device)
        y_test = Variable(x_test)
            
        prediction = model(x_test.float())
            
        for k in range(l, r):
            if (prediction[k-l][0] > prediction[k-l][1]) and (y_test[k] == 0):
                true = true + 1
            elif (prediction[k-l][0] > prediction[k-l][1]) and (y_test[k] == 1):
                false = false + 1
            elif (prediction[k-l][0] <= prediction[k-l][1]) and (y_test[k] == 1):
                true = true + 1
            else:
                false = false + 1
        
    return true / (true + false)
