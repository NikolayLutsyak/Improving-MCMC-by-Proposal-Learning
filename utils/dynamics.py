import torch
import numpy as np
from time import time as localtime

def safe_exp(x, name=None):
    return torch.exp(x)

class Dynamics(object):
    def __init__(self,
                 x_dim,
                 energy_function,
                 T=5,
                 eps=0.1,
                 hmc=False,
                 net_factory=None,
                 encoder_sampler=None,
                 size1=10,
                 size2=10,
                 model_paths=None):
        self.hmc = hmc
        self.x_dim = x_dim
        self.energy_function = energy_function
        self.T = T
        self.eps = torch.tensor(eps).cuda(1)
        if hmc:
            z = lambda x, *args, **kwargs: torch.zeros_like(x).cuda(1)
            self.XNet = lambda inp: [torch.zeros_like(inp[0]).cuda(1) for t in range(3)]
            self.VNet = lambda inp: [torch.zeros_like(inp[0]).cuda(1) for t in range(3)]
        else:
            self.XNet = net_factory(x_dim, size1, size2, encoder=encoder_sampler)
            self.VNet = net_factory(x_dim, size1, size2, encoder=encoder_sampler)
            if model_paths is not None:
                self.XNet.load_state_dict(torch.load(model_paths[0]))
                self.VNet.load_state_dict(torch.load(model_paths[1]))

            self.XNet.cuda(1)
            self.VNet.cuda(1)
        self._init_mask()

    def _init_mask(self):
        mask_per_step = []
        for t in range(self.T):
            ind = np.random.permutation(np.arange(self.x_dim))[
                :int(self.x_dim / 2)]
            m = np.zeros((self.x_dim,))
            m[ind] = 1
            mask_per_step.append(m)
        self.mask = torch.FloatTensor(np.stack(mask_per_step)).cuda(1)

    def _get_mask(self, t):
        m = self.mask[t]
        return m, 1 - m

    def _format_time(self, t, tile):
        trig_t = torch.Tensor([torch.cos(torch.tensor(
            2 * np.pi * t / self.T)), torch.sin(torch.tensor(2 * np.pi * t / self.T))])
        return trig_t.repeat(tile, 1)

    def _forward_step(self, x, v, step, aux=None):
        t = self._format_time(step, tile=x.size()[0]).cuda(1)

        grad1 = self.grad_energy(x.data, aux)
        S1 = self.VNet([x, grad1, t, aux.cuda(1)])  # rewrite for torch

        sv1 = 0.5 * self.eps * S1[0]  # rewrite for torch
        tv1 = S1[1]  # rewrite for torch
        fv1 = self.eps * S1[2]  # rewrite for torch

        v_h = v.cuda(1) * safe_exp(sv1, name='sv1F') + 0.5 * self.eps * \
            (- safe_exp(fv1, name='fv1F') * grad1.cuda(1)) + tv1

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, (m * x), t, aux])  # rewrite for torch

        sx1 = (self.eps * X1[0])  # rewrite for torch
        tx1 = X1[1]  # rewrite for torch
        fx1 = self.eps * X1[2]  # rewrite for torch

        y = m * x + mb * (x * safe_exp(sx1, name='sx1F') +
                          self.eps * (safe_exp(fx1, name='fx1F') * v_h + tx1))

        X2 = self.XNet([v_h, mb * y, t, aux])  # rewrite for torch

        sx2 = (self.eps * X2[0])  # rewrite for torch
        tx2 = X2[1]  # rewrite for torch
        fx2 = self.eps * X2[2]  # rewrite for torch

        x_o = mb * y + m * (y * safe_exp(sx2, name='sx2F') +
                            self.eps * (safe_exp(fx2, name='fx2F') * v_h + tx2))

        # rewrite for torch
        S2 = self.VNet([x_o, self.grad_energy(x_o.data, aux), t, aux])
        sv2 = (0.5 * self.eps * S2[0])  # rewrite for torch
        tv2 = S2[1]  # rewrite for torch
        fv2 = self.eps * S2[2]  # rewrite for torch

        grad2 = self.grad_energy(x_o.data, aux)
        v_o = v_h * safe_exp(sv2, name='sv2F') + 0.5 * self.eps * \
            (- safe_exp(fv2, name='fv2F') * grad2 + tv2)

        log_jac_contrib = torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

        return x_o, v_o, log_jac_contrib

    def _backward_step(self, x_o, v_o, step, aux=None):
        t = self._format_time(step, tile=x_o.size()[0]).cuda(1)

        grad1 = self.grad_energy(x_o.data, aux)

        S1 = self.VNet([x_o.cuda(1), grad1, t, aux])

        sv2 = (-0.5 * self.eps * S1[0])
        tv2 = S1[1]
        fv2 = self.eps * S1[2]

        v_h = (v_o.cuda(1) - 0.5 * self.eps * (- safe_exp(fv2, name='fv2B')
                                       * grad1 + tv2)) * safe_exp(sv2, name='sv2B')

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, mb * x_o, t, aux])

        sx2 = (-self.eps * X1[0])
        tx2 = X1[1]
        fx2 = self.eps * X1[2]

        y = mb * x_o + m * safe_exp(sx2, name='sx2B') * (x_o -
                                                         self.eps * (safe_exp(fx2, name='fx2B') * v_h + tx2))

        X2 = self.XNet([v_h, m * y, t, aux])

        sx1 = (-self.eps * X2[0])
        tx1 = X2[1]
        fx1 = self.eps * X2[2]

        x = m * y + mb * safe_exp(sx1, name='sx1B') * (y -
                                                       self.eps * (safe_exp(fx1, name='fx1B') * v_h + tx1))

        grad2 = self.grad_energy(x.data, aux)
        S2 = self.VNet([x, grad2, t, aux])

        sv1 = (-0.5 * self.eps * S2[0])
        tv1 = S2[1]
        fv1 = self.eps * S2[2]

        v = safe_exp(sv1, name='sv1B') * (v_h - 0.5 * self.eps *
                                          (-safe_exp(fv1, name='fv1B') * grad2 + tv1))

        return x, v, torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

    def energy(self, x, aux=None):
        if aux is not None:
            return self.energy_function(x, aux)
        else:
            return self.energy_function(x)

    def grad_energy(self, x, aux=None):
        x.requires_grad = True
        en = self.energy(x, aux)
        grad_en = torch.autograd.grad(torch.sum(en), x)[0]
        x.requires_grad = False
        return grad_en


       
    def forward(self, x, init_v=None, aux=None, log_path=False, log_jac=False):
        if init_v is None:
            v = torch.randn(x.size()).cuda(1)
        else:
            v = init_v
        j = torch.zeros(x.size()[0]).cuda(1)

        x_old = x
        v_old = v
        t = 0
        while t < self.T:
            x, v, log_jac_ = self._forward_step(x, v, t, aux)
            t += 1
            j += log_jac_
        
        if log_jac:
            return x, v, j

        return x, v, self.p_accept(x_old, v_old.cuda(1), x, v.cuda(1), j, aux)


    def backward(self, x, init_v=None, aux=None, log_jac=False):
        if init_v is None:
            v = torch.randn(x.size()).cuda(1)
        else:
            v = init_v
        j = torch.zeros(x.size()[0]).cuda(1)

        x_old = x
        v_old = v
        t = 0
        while t < self.T:
            x, v, log_jac_ = self._backward_step(x, v, self.T - t - 1, aux.cuda(1))
            t += 1
            j += log_jac_
        
        if log_jac:
            return x, v, j

        return x, v, self.p_accept(x_old, v_old.cuda(1), x, v.cuda(1), j, aux)
    
    def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
        e_new = self.hamiltonian(x1, v1, aux)
        e_old = self.hamiltonian(x0, v0, aux)

        v = e_old - e_new + log_jac
        p = torch.exp(torch.min(v, torch.tensor(0.0).cuda(1)))

        return torch.where(torch.isfinite(p), p, torch.zeros_like(p))

    def kinetic(self, v):
        return 0.5 * torch.sum(v**2, dim=1)

    def hamiltonian(self, x, v, aux=None):
        return self.energy(x, aux) + self.kinetic(v)
