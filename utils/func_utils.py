import torch
import os
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import openml
from torch.utils.data import TensorDataset, DataLoader


def vae_loss(mu, log_sigma, logits, x):
    kl = normal_kl(mu, torch.exp(log_sigma), torch.tensor(0.), torch.tensor(1.))
    bce = torch.sum(torch.nn.functional.binary_cross_entropy_with_logits(logits, x, reduction='none'), 1)
    elbo = torch.mean(kl + bce)
    return elbo


def binarize(x):
    assert(x.max() <= 1.)
    return (np.random.random(x.shape) < x).astype(np.float32)

def gen_data(batch_size=10):
    if os.path.exists('mnist_data.npy'):
        X = np.load('mnist_data.npy')
    else:
        mnist = openml.fetch_openml('mnist_784')
        X = mnist['data']
        X.dump('mnist_data.npy')
    
    X_test = X[-10000:]
    X_train = X[:60000]

    X_test = binarize(X_test/255.)
    X_train = binarize(X_train/255.)

    dataset_test = TensorDataset(torch.from_numpy(X_test))
    dataset_train = TensorDataset(torch.from_numpy(X_train))

    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    return test_loader, train_loader


def accept(x_i, x_p, p):
    assert x_i.shape == x_p.shape

    dN, dX = x_i.shape
    u = np.random.uniform(size=(dN,))
    m = (p - u >= 0).astype('int32')[:, None]
    return x_i * (1 - m) + x_p * m


def autocovariance(X, tau=0):
    dT, dN, dX = np.shape(X)
    s = 0.
    for t in range(dT - tau):
        x1 = X[t, :, :]
        x2 = X[t+tau, :, :]

        s += np.sum(x1 * x2) / dN

    return s / (dT - tau)

def get_log_likelihood(X, gaussian):
    m = multivariate_normal(mean=gaussian.mu, cov=gaussian.sigma)
    return m.logpdf(X).mean()


def t_accept(x, Lx, px):
    mask = (px - torch.rand(px.shape) >= 0.)
    return torch.where(mask, Lx, x)


def normal_kl(q_mu, q_sigma, p_mu, p_sigma):
    q_entropy = 0.5 + torch.log(q_sigma)
    q_p_cross_entropy = 0.5 * (q_sigma / p_sigma)**2 + 0.5 * ((q_mu - p_mu) / p_sigma)**2 + torch.log(p_sigma)
    q_p_kl = torch.sum(-q_entropy + q_p_cross_entropy, -1)
    return q_p_kl

def acl_spectrum(X, scale):
    n = X.shape[0]
    return np.array([autocovariance(X / scale, tau=t) for t in range(n-1)])


def ESS(A):
    A = A * (A > 0.05)
    return 1. / (1. + 2 * np.sum(A[1:]))
