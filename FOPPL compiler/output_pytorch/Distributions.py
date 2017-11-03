from DistributionClass.agnostic_tensor import *
import numpy as np
from torch.autograd import Variable
import torch


def VariableCast(value):
    if isinstance(value, torch.autograd.variable.Variable):
        return value
    elif torch.is_tensor(value):
        return Variable(value)
    else:
        return Variable(torch.Tensor([value]))


class Normal():
    """Normal random variable"""
    def __init__(self, mean, variance):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable [batch_size, num_particles]
            variance: Tensor/Variable [batch_size, num_particles]
        """
        self.mean = VariableCast(mean)
        self.variance = VariableCast(variance)


    def sample(self, num_samples = 1):
        # x = Variable(torch.randn(1), requires_grad = True)
        # #sample = Variable.add(Variable.mul(x, Variable.sqrt(self.variance), self.mean))
        # sample = x * torch.sqrt(self.variance) + self.mean
        # sample.retain_grad()

        x = torch.randn(1)
        sample = Variable(x * torch.sqrt(self.variance.data) + self.mean.data, requires_grad = True)
        return sample #.detach()


    def logpdf(self, value):
        mean = self.mean
        var = self.variance
        value = VariableCast(value)
        # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
        return (-0.5 * Variable.pow(value - mean, 2) / var - 0.5 * Variable.log(2 * var * np.pi))


