import torch
from ..distributions.distribution_wrapper import TorchDistribution
from ..utils.core import VariableCast as vc

class Poisson(TorchDistribution):
    r""" Creates a Poisson distribution parameterized by `rate`, the rate parameter.
Samples are nonnegative integers, with a pmf given by$rate^k e^{-rate}/k!$
    Example::
        >>> m = Poisson(torch.Tensor([4]))
        >>> m.sample()
         3
        [torch.LongTensor of size 1]
    Args:
        rate (Number, Tensor or Variable): the rate parameter
"""
    def __init__(self, lam):
        self.rate= vc(lam)
        torch_dist = torch.distributions.Poisson(rate=self.rate)
        super(Poisson, self).__init__(torch_dist)