import numpy as np
import torch
from time import time


def propose(x, dynamics, init_v=None, aux=None, do_mh_step=False, log_jac=False):
    if dynamics.hmc:
        Lx, Lv, px = dynamics.forward(x, init_v=init_v, aux=aux)
        return Lx, Lv, px, [t_accept(x, Lx, px)]
    else:
        # sample mask for forward/backward
        mask = torch.randint(high=2, size = (x.shape[0], 1)).float().cuda(1)
        Lx1, Lv1, px1 = dynamics.forward(x, aux=aux, log_jac=log_jac)
        Lx2, Lv2, px2 = dynamics.backward(x, aux=aux, log_jac=log_jac)

        Lx = mask * Lx1 + (1 - mask) * Lx2

        Lv = None
        if init_v is not None:
            Lv = mask * Lv1 + (1 - mask) * Lv2

        px = mask.squeeze(1) * px1 + (1 - mask).squeeze(1) * px2

        outputs = []

        if do_mh_step:
            outputs.append(t_accept(x, Lx, px))

        return Lx, Lv, px, outputs

def t_accept(x, Lx, px):
    mask = (px - torch.rand(px.shape).cuda(1) >= 0.)
    return torch.where(mask.expand(x.shape[1], x.shape[0]).transpose(1, 0), Lx, x)

def chain_operator(init_x, dynamics, nb_steps, aux=None, init_v=None, do_mh_step=False):
    if not init_v:
        init_v = torch.randn(init_x.shape)
    t = torch.tensor(0.)
    log_jac = torch.zeros((init_x.shape[0], ))
   
    while t < nb_steps.float():
        Lx, Lv, px, _ = propose(init_x, dynamics, init_v=v, aux=aux, log_jac=True, do_mh_step=False)
        log_jac = log_jac + px
        t = t + 1

    p_accept = dynamics.p_accept(init_x, init_v, final_x, final_v, log_jac, aux=aux)

    outputs = []
    if do_mh_step:
        outputs.append(t_accept(init_x, final_x, p_accept))

    return final_x, final_v, p_accept, outputs
