from utils.dynamics import Dynamics
from utils.sampler import propose
from model import Net
from torch.optim.lr_scheduler import ExponentialLR
from itertools import chain
import numpy as np
import torch

def compute_loss(x, z, dynamics):
    Lx, _, px, output = propose(x, dynamics, do_mh_step=True)
    Lz, _, pz, _ = propose(z, dynamics, do_mh_step=False)

    v1 = (torch.sum((x - Lx)**2, dim=1) * px) + 1e-4
    v2 = (torch.sum((z - Lz)**2, dim=1) * pz) + 1e-4
    scale = 0.1

    loss = (scale * (torch.mean(1.0 / v1) + torch.mean(1.0 / v2)) +
                        (-torch.mean(v1) - torch.mean(v2)) / scale)
    return loss, output[0]

def train(distribution, x_dim):
    dynamics = Dynamics(x_dim, distribution.get_energy_function(), T=10, eps=0.1, net_factory=Net)

    n_steps = 5000
    n_samples = 200

    x = torch.randn(size=(n_samples, x_dim)).float()
    optimizer = torch.optim.Adam(chain(dynamics.XNet.parameters(), dynamics.VNet.parameters()), lr=0.001)
    lr_sheduler = ExponentialLR(optimizer, 0.96**(-1/1000) )

    for i in range(n_steps):
        lr_sheduler.step()
        z = torch.randn(x.shape)
        loss, x = compute_loss(x, z, dynamics)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
