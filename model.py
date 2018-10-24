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
    def __init__(self, x_dim, size1=10, size2=10, encoder=None):
        super(Net, self).__init__()
         
        self.encoder = encoder
        self.linear_x = nn.Linear(x_dim, size1)
        self.linear_v = nn.Linear(x_dim, size1)
        self.linear_t = nn.Linear(2, size1)
        
        self.linear = nn.Linear(size1, size2)
        
        self.linear_Q = nn.Linear(size2, x_dim)
        self.linear_S = nn.Linear(size2, x_dim)
        self.linear_T = nn.Linear(size2, x_dim)
        
        self.relu = nn.ReLU()
        self.scaletanh_Q = ScaleTanh()
        self.scaletanh_S = ScaleTanh()
        
               
    def forward(self, input_):
        v, x, t, aux = input_
        out_enc = self.encoder(aux) if self.encoder is not None else 0
        out_lin_x = self.linear_x(x)
        out_lin_v = self.linear_v(v)
        out_lin_t = self.linear_t(t)
        
        out_sum = out_lin_x + out_lin_v + out_lin_t + out_enc
        
        out_relu1  = self.relu(out_sum)
        out_lin = self.linear(out_relu1)
        out_relu2  = self.relu(out_lin)
        
        out_Q = self.scaletanh_Q(self.linear_Q(out_relu2))
        out_S = self.scaletanh_S(self.linear_S(out_relu2))
        out_T = self.linear_T(out_relu2)
        
        return out_S, out_Q, out_T 
    
class Encoder(nn.Module):
    def __init__(self, in_features, out_features, mid_features):
        super(Encoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, mid_features)
        self.linear2 = nn.Linear(mid_features, mid_features)
        self.linear3 = nn.Linear(mid_features, out_features)
        self.linear4 = nn.Linear(mid_features, out_features)
        
        self.softplus = nn.Softplus()    
        self.seq = nn.Sequential(self.linear1,
                                 self.softplus,
                                 self.linear2, 
                                 self.softplus)
               
    def forward(self, x):
        out_seq = self.seq(x)
        out_lin1 = self.linear3(out_seq)
        out_lin2 = self.linear4(out_seq)
        return out_lin1, out_lin2
    
class Decoder(nn.Module):
    def __init__(self, in_features, out_features, mid_features):
        super(Decoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, mid_features)
        self.linear2 = nn.Linear(mid_features, mid_features)
        self.linear3 = nn.Linear(mid_features, out_features)
        
        self.softplus = nn.Softplus()    
        self.seq = nn.Sequential(self.linear1,
                                 self.softplus,
                                 self.linear2, 
                                 self.softplus,
                                 self.linear3)
               
    def forward(self, x):
        return self.seq(x)        
    
class VAE(nn.Module):
    def __init__(self, in_features, latent_dim, mid_features):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_features, latent_dim, mid_features)
        self.decoder = Decoder(latent_dim, in_features, mid_features)
               
    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        noise = torch.randn(mu.shape)
        latent_q = mu + noise.cuda(1) * torch.exp(log_sigma)
        self.z = latent_q
        logits = self.decoder(latent_q)
        return logits, mu, log_sigma
    
    def sample(self, z):
        return torch.sigmoid(self.decoder(z))
