import torch
import numpy as np


class Dynamics(object):
    def __init__(self,
                 x_dim,
                 energy_function,
                 T=25,
                 eps=0.1,
                 net_factory=None):
        self.x_dim = x_dim
        self.energy_function = energy_function
        self.T = T
        self.Xnet = net_factory
        self.VNet = net_factory
        self._init_mask()


    def _init_mask(self):
        mask_per_update = []
        for t in range(self.T):
            ind = np.random.permutation(np.arange(self.x_dim))[:int(self.x_dim / 2)]
            m = np.zeros((self.x_dim,))
            m[ind] = 1
            mask_per_step.append(m)
        self.mask = torch.FloatTensor(np.stack(mask_per_step))


    def _get_mask(self, t):
        m = self.mask[t]
        return m, 1 - m


    def _format_time(self, t, tile):
        trig_t = torch.Tensor([torch.cos(2 * np.pi * t / self.T), torch.sin(2 * np.pi * t / self.T)])
        return trig_t.repeat(tile, 1)


    def _forward_step(self, x, v, step):
        t = self._format_time(step, tile=x.size()[0])

        grad1 = self.grad_energy(x)
        S1 = self.VNet([x, grad1, t]) # rewrite for torch

        sv1 = 0.5 * self.eps * S1[0] # rewrite for torch
        tv1 = S1[1] # rewrite for torch
        fv1 = self.eps * S1[2] # rewrite for torch

        v_h = v * safe_exp(sv1, name='sv1F') + 0.5 * self.eps * (- safe_exp(fv1, name='fv1F') * grad1) + tv1)

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, m * x, t, aux]) # rewrite for torch

        sx1 = (self.eps * X1[0]) # rewrite for torch
        tx1 = X1[1] # rewrite for torch
        fx1 = self.eps * X1[2] # rewrite for torch

        y = m * x + mb * (x * safe_exp(sx1, name='sx1F') + self.eps * (safe_exp(fx1, name='fx1F') * v_h + tx1))

        X2 = self.XNet([v_h, mb * y, t]) # rewrite for torch

        sx2 = (self.eps * X2[0]) # rewrite for torch
        tx2 = X2[1] # rewrite for torch
        fx2 = self.eps * X2[2] # rewrite for torch

        x_o = mb * y + m * (y * safe_exp(sx2, name='sx2F') + self.eps * (safe_exp(fx2, name='fx2F') * v_h + tx2))

        S2 = self.VNet([x_o, self.grad_energy(x_o), t]) # rewrite for torch
        sv2 = (0.5 * self.eps * S2[0]) # rewrite for torch
        tv2 = S2[1] # rewrite for torch
        fv2 = self.eps * S2[2] # rewrite for torch

        grad2 = self.grad_energy(x_o)
        v_o = v_h * safe_exp(sv2, name='sv2F') + 0.5 * self.eps * (- safe_exp(fv2, name='fv2F') * grad2 + tv2)

        log_jac_contrib = torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)

        return x_o, v_o, log_jac_contrib


    def _backward_step(self, x_o, v_o, step, aux=None):
        t = self._format_time(step, tile=x_o.size()[0])

        grad1 = self.grad_energy(x_o)

        S1 = self.VNet([x_o, grad1, t])

        sv2 = (-0.5 * self.eps * S1[0])
        tv2 = S1[1]
        fv2 = self.eps * S1[2]

        v_h = (v_o - 0.5 * self.eps * (- safe_exp(fv2, name='fv2B') * grad1 + tv2)) * safe_exp(sv2, name='sv2B')

        m, mb = self._get_mask(step)

        X1 = self.XNet([v_h, mb * x_o, t])

        sx2 = (-self.eps * X1[0])
        tx2 = X1[1]
        fx2 = self.eps * X1[2]

        y = mb * x_o + m * safe_exp(sx2, name='sx2B') * (x_o - self.eps * (safe_exp(fx2, name='fx2B') * v_h + tx2))

        X2 = self.XNet([v_h, m * y, t])

        sx1 = (-self.eps * X2[0])
        tx1 = X2[1]
        fx1 = self.eps * X2[2]

        x = m * y + mb * safe_exp(sx1, name='sx1B') * (y - self.eps * (safe_exp(fx1, name='fx1B') * v_h + tx1))

        grad2 = self.grad_energy(x)
        S2 = self.VNet([x, grad2, t])

        sv1 = (-0.5 * self.eps * S2[0])
        tv1 = S2[1]
        fv1 = self.eps * S2[2]

        v = safe_exp(sv1, name='sv1B') * (v_h - 0.5 * self.eps * (-safe_exp(fv1, name='fv1B') * grad2 + tv1))

        return x, v, torch.sum(sv1 + sv2 + mb * sx1 + m * sx2, dim=1)


    def energy(self, x):
        return self.energy_function(x)


    def grad_energy(self, x):
        y = torch.Tensor(x, requires_grad=True)
        en = self.energy(y)
        en.backward()
        return y.grad


    def forward(self, x, init_v=None, aux=None, log_path=False, log_jac=False):
        if init_v is None:
          v = torch.randn(x.size())
        else:
          v = init_v

        # dN = x.size()[0]
        # t = tf.constant(0., dtype=TF_FLOAT)
        j = torch.zeros(x.size()[0])

        t = 0
        while t < self.T:
            X, V, log_jac_ = self._forward_step(x, v, t)
            t += 1
        log_jac_ = log_jac_ + j

        if log_jac:
          return X, V, log_jac_

        return X, V, self.p_accept(x, v, X, V, log_jac_)


    def backward(self, x, init_v=None, aux=None, log_jac=False):
        if init_v is None:
          v = torch.randn(x.size())
        else:
          v = init_v

        # dN = tf.shape(x)[0]
        # t = tf.constant(0., name='step_backward', dtype=TF_FLOAT)
        j = torch.zeros(x.size()[0])

        t = 0
        while t < self.T:
            X, V, log_jac_ = self._backward_step(x, v, self.T - t - 1)
            t += 1
        log_jac_ = log_jac_ + j

        if log_jac:
          return X, V, log_jac_

        return X, V, self.p_accept(x, v, X, V, log_jac_)


    def p_accept(self, x0, v0, x1, v1, log_jac, aux=None):
        e_new = self.hamiltonian(x1, v1)
        e_old = self.hamiltonian(x0, v0)

        v = e_old - e_new + log_jac
        p = torch.exp(torch.min(v, 0.0))

        return torch.where(torch.isfinite(p), p, torch.zeros_like(p))


    def kinetic(self, v):
        return 0.5 * torch.sum(tf.square(v), dim=1)


    def hamiltonian(self, x, v, aux=None):
        return self.energy(x) + self.kinetic(v)