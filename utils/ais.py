import numpy as np
import torch

from utils.dynamics import Dynamics


def ais_estimate(init_energy, final_energy, anneal_steps, initial_x, aux=None,
                 step_size=0.5, leapfrogs=25, x_dim=5, num_splits=1,
                 refresh=False, refreshment=0.1):
    beta = torch.linspace(0., 1., anneal_steps+1)[1:].cuda(1)
    beta_diff = beta[1] - beta[0]
    refreshment = torch.tensor(refreshment).cuda(1)

    def body(a, beta):
        def curr_energy(z, aux=None):
            return ((1-beta) * init_energy(z) + (beta) * final_energy(z, aux=aux))
        last_x = a[1]
        w = a[2]
        v = a[3]
        if refresh:
            refreshed_v = v * torch.sqrt(1-refreshment) + torch.randn(v.shape).cuda(1) * torch.sqrt(refreshment)
        else:
            refreshed_v = torch.randn(v.shape).cuda(1)
        w = w + beta_diff * (- final_energy(last_x, aux=aux) + init_energy(last_x, aux=aux))

        dynamics = Dynamics(x_dim, energy_function=curr_energy, eps=step_size, hmc=True, T=leapfrogs)
        Lx, Lv, px = dynamics.forward(last_x, aux=aux, init_v=refreshed_v)

        mask = (px - torch.rand(px.shape).cuda(1) >= 0.)
        mask = mask.expand(Lx.shape[1], Lx.shape[0]).transpose(1, 0)
        updated_x = torch.where(mask, Lx, last_x)
        updated_v = torch.where(mask, Lv, -Lv)

        return (px.data, updated_x.data, w.data, updated_v.data)

    a = (torch.zeros_like(initial_x[:, 0]).cuda(1), initial_x, 
         torch.zeros_like(initial_x[:, 0]).cuda(1), torch.randn(initial_x.shape).cuda(1))

    for b in beta:
        a = body(a, b)
    alpha, x, w, _ = a

    def logmeanexp(z): 
        return torch.logsumexp(z, 0) - torch.log(torch.tensor(z.shape[0]).float().cuda(1))

    if num_splits == 1:
        return logmeanexp(w), torch.mean(alpha)
    
    list_w = torch.split(w, num_splits, dim=0)
    return (torch.sum(torch.stack(list(map(logmeanexp, list_w)), dim=0), 0), torch.mean(alpha))
