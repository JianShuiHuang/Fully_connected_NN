import torch
import torch.nn as nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class fully_connected(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(fully_connected, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        ##layer 1
        self.linear_1 = nn.Linear(in_features=in_feature, out_features=self.out_feature, bias=True)
        self.activ_1 = nn.Sigmoid()
        
        ##layer 2
        self.linear_2 = nn.Linear(in_features=self.out_feature, out_features=2, bias=False)
                
    ##forwarding and backpropagation
    def forward(self, x):
        y = self.linear_1(x)
        y = self.activ_1(y)
	
	y = y.view(y.size(0), -1)
        y = self.linear_2(y)
		
        return y
