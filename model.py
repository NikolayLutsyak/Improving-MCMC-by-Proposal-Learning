import torch
from torch import nn
from torch.nn.parameter import Parameter

class ScaleTanh(nn.Module):
    def __init__(self):
        super(ScaleTanh, self).__init__()
        self.scale = Parameter(torch.tensor([0.]))
        
    def forward(self, x):
        return torch.exp(self.scale) * torch.tanh(x)


class Net(nn.Module):
    def __init__(self, x_dim):
        super(Net, self).__init__()
                
        self.linear_x = nn.Linear(x_dim, 10)
        self.linear_v = nn.Linear(x_dim, 10)
        self.linear_t = nn.Linear(2, 10)
        
        self.linear = nn.Linear(10, 10)
        
        self.linear_Q = nn.Linear(10, x_dim)
        self.linear_S = nn.Linear(10, x_dim)
        self.linear_T = nn.Linear(10, x_dim)
        
        self.relu = nn.ReLU()
        self.scaletanh_Q = ScaleTanh()
        self.scaletanh_S = ScaleTanh()
        
               
    def forward(self, input_):
        v, x, t, _ = input_
        out_lin_x = self.linear_x(x)
        out_lin_v = self.linear_v(v)
        out_lin_t = self.linear_t(t)
        
        out_sum = out_lin_x + out_lin_v + out_lin_t
        
        out_relu1  = self.relu(out_sum)
        out_lin = self.linear(out_relu1)
        out_relu2  = self.relu(out_lin)
        
        out_Q = self.scaletanh_Q(self.linear_Q(out_relu2))
        out_S = self.scaletanh_S(self.linear_S(out_relu2))
        out_T = self.linear_T(out_relu2)
        
        return out_S, out_Q, out_T 