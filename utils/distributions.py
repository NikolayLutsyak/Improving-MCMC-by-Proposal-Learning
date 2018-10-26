import numpy as np
import torch
from scipy.stats import multivariate_normal, ortho_group
import collections


def quadratic_gaussian(x, mu, S):
  return torch.diag(0.5 * (x - mu) @ S @ (x - mu).transpose(0, 1))


class Gaussian(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.i_sigma = np.linalg.inv(np.copy(sigma))

    def get_energy_function(self):
        def fn(x, *args, **kwargs):
            S = torch.Tensor(self.i_sigma.astype('float32'))
            mu = torch.Tensor(self.mu.astype('float32'))
            return quadratic_gaussian(x, mu, S)
        return fn

    def get_samples(self, n):
        C = np.linalg.cholesky(self.sigma)
        X = np.random.randn(n, self.sigma.shape[0])
        return X.dot(C.T)


class TiltedGaussian(Gaussian):
    def __init__(self, dim, log_min, log_max):
        self.R = ortho_group.rvs(dim)
        self.diag = np.diag(np.exp(np.log(10.) * np.random.uniform(log_min, log_max, size=(dim,)))) + 1e-8 * np.eye(dim)
        S = self.R.T.dot(self.diag).dot(self.R)
        self.dim = dim
        Gaussian.__init__(self, np.zeros((dim,)), S)

    def get_samples(self, n):
        X = np.random.randn(200, self.dim)
        X = X.dot(np.sqrt(self.diag))
        X = X.dot(self.R)
        return X


class RoughWell(object):
    def __init__(self, dim, eps, easy=False):
        self.dim = dim
        self.eps = eps
        self.easy = easy

    def get_energy_function(self):
        def fn(x, *args, **kwargs):
            n = torch.sum(x**2, 1)
            if not self.easy:
                return 0.5 * n + self.eps * torch.sum(torch.cos(x / (self.eps * self.eps)), 1)
            else:
                return 0.5 * n + self.eps * torch.sum(torch.cos(x / self.eps), 1)
        return fn

    def get_samples(self, n):
        # we can approximate by a gaussian for eps small enough
        return np.random.randn(n, self.dim)


class GMM(object):
    def __init__(self, mus, sigmas, pis):
        assert len(mus) == len(sigmas)
        assert sum(pis) == 1.0

        self.mus = [torch.FloatTensor(mu) for mu in mus]
        self.sigmas = sigmas
        self.pis = pis

        self.nb_mixtures = len(pis)

        self.k = mus[0].shape[0]

        self.i_sigmas = []
        self.constants = []

        for i, sigma in enumerate(sigmas):
            self.i_sigmas.append(torch.FloatTensor(np.linalg.inv(sigma).astype('float32')))
            det = np.sqrt((2 * np.pi) ** self.k * np.linalg.det(sigma)).astype('float32')
            self.constants.append(torch.FloatTensor([(pis[i] / det).astype('float32')]))

    def get_energy_function(self):
        def fn(x):
            V = torch.cat([
            torch.unsqueeze(-quadratic_gaussian(x, self.mus[i], self.i_sigmas[i])
                     + torch.log(self.constants[i]), 1)
            for i in range(self.nb_mixtures)
            ], dim=1)

            return -torch.logsumexp(V, dim=1)
        return fn

    def get_samples(self, n):
        categorical = np.random.choice(self.nb_mixtures, size=(n,), p=self.pis)
        counter_samples = collections.Counter(categorical)

        samples = []

        for k, v in counter_samples.items():
            samples.append(np.random.multivariate_normal(self.mus[k], self.sigmas[k], size=(v,)))

        samples = np.concatenate(samples, axis=0)

        np.random.shuffle(samples)

        return samples
