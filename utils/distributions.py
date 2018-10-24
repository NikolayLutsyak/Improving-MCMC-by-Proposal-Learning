import numpy as np
import torch


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
