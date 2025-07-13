import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *

from timeit import default_timer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.optim import Adam
import pickle

from scipy.io import savemat
from scipy import io
torch.manual_seed(0)
np.random.seed(0)

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
set_seed(0)
import sys

class FNN(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, layer_sizes):
        super(FNN, self).__init__()

        layers = []
        for i in range(1, len(layer_sizes) - 1):
            layers.append(nn.Linear(in_features=layer_sizes[i - 1], out_features=layer_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=layer_sizes[-2], out_features=layer_sizes[-1]))

        self.denses = nn.ModuleList(layers)

    def forward(self, inputs):
        y = inputs
        for f in self.denses:
            y = f(y)
        return y
    
class DeepONet(nn.Module):
    def __init__(self, layer_size_branch,layer_size_trunk):
        super(DeepONet, self).__init__()

        self.layer_size_branch = layer_size_branch
        #self.layer_size_bc = layer_size_bc
        self.layer_size_trunk = layer_size_trunk
        
        #self.b1=nn.Linear(3, 1)
        #self.b2=nn.Linear(3, 1)
        

        self.branch_net = FNN(self.layer_size_branch)
        self.trunk_net = FNN(self.layer_size_trunk)

        self.bias_last = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x_branch,x_trunk):

        y_branch = self.branch_net(x_branch)
       
     
        y_trunk = self.trunk_net(x_trunk)
        
        Y = torch.einsum("bi,bni->bn", y_branch, y_trunk)
        Y += self.bias_last
        return Y

    
    
################################################################
# configs
################################################################

ntrain = 500
ntest = 100

batch_size = 20
learning_rate= 0.0001
epochs = 5000


################################################################
# load data and data normalization
################################################################
u_all = scipy.io.loadmat('u_all.mat')['u_all']
f_all = scipy.io.loadmat('f_all.mat')['f_all']
xy_all = scipy.io.loadmat('xy_all.mat')['xy_all']
N_xy=51*51

u_all=u_all.reshape(600,N_xy)
f_all=f_all.reshape(600,N_xy)

input_u = torch.tensor(u_all, dtype=torch.float32)
input_f = torch.tensor(f_all, dtype=torch.float32)
input_xy= torch.tensor(xy_all, dtype=torch.float32)

train_f = input_f[:ntrain]
test_f = input_f[-ntest:]
train_u = input_u[:ntrain]
test_u = input_u[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]


print(train_f.shape, train_u.shape, train_xy.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_u, train_f, train_xy), batch_size=batch_size, shuffle=True) 
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_u, test_f, test_xy), batch_size=batch_size, shuffle=False) 



################################################################
# training and evaluation
################################################################
layer_size_branch=[N_xy,256,256,256,64]
layer_size_trunk=[2,256,256,256,64]
batch_size = 20
learning_rate = 0.001
epochs = 5000
step_size = 1000
gamma = 0.5

model = DeepONet(layer_size_branch,layer_size_trunk).cuda()
print(count_params(model))


optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

#optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20, history_size=10, line_search_fn="strong_wolfe")


myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for u, f, xy in train_loader:
        u, f, xy= u.to(device), f.to(device), xy.to(device)
        optimizer.zero_grad()
        out = model(f,xy).reshape(batch_size,N_xy)
        #out_b=model(f, bc,xyb).reshape(batch_size,N_bc)
        loss_u =  myloss(out.view(batch_size, -1), u.view(batch_size, -1))
        #loss_g=   myloss(out_b.view(batch_size, -1), ub.view(batch_size, -1))
        loss = loss_u
        loss.backward()
       
        optimizer.step()
        train_l2 += loss.item()

    # 更新学习率调度器
    scheduler.step()

    model.eval()
    
    with torch.no_grad():
        for u, f, xy in test_loader:
            u, f, xy = u.cuda(), f.cuda(), xy.cuda()
            out = model(f, xy).reshape(batch_size,N_xy)
        
    train_l2 /= 1000
    
    t2 = default_timer()
    print(ep, t2 - t1, train_l2)

    if ep % 100 == 0:
        torch.save(model, 'possion')
        XY = xy[-1].squeeze().detach().cpu().numpy()
        truth = u[-1].squeeze().detach().cpu().numpy()
        pred = out[-1].squeeze().detach().cpu().numpy()
        
        fig = plt.figure(figsize=(9, 3))
        plt.subplots_adjust(wspace=0.2)

        # 第一列: 显示实际值
        plt.subplot(1, 3, 1)
        plt.scatter(XY[:, 0], XY[:, 1], c=truth, cmap='viridis')
        plt.title('Truth')
        plt.colorbar(fraction=0.045)
        # 第二列: 显示预测值
        plt.subplot(1, 3, 2)
        plt.scatter(XY[:, 0], XY[:, 1], c=pred, cmap='viridis')
        plt.title('Predicted')
        plt.colorbar(fraction=0.045)
        
        # 第三列: 显示误差
        plt.subplot(1, 3, 3)
        plt.scatter(XY[:, 0], XY[:, 1], c=pred - truth, cmap='viridis')
        plt.title('Error')
        plt.colorbar(fraction=0.045)

        plt.tight_layout()
        plt.savefig('output1.png')
        plt.show()


