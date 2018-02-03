import torch

from pyfo.distributions.distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Uniform(TorchDistribution):
    r"""
    Generates uniformly distributed random samples from the half-open interval
    `[low, high)`.

    Example::

        >>> m = Uniform(torch.Tensor([0.0]), torch.Tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
         2.3418
        [torch.FloatTensor of size 1]

    Args:
        low (float or Tensor or Variable): lower range (inclusive).
        high (float or Tensor or Variable): upper range (exclusive).
    """
    def __init__(self, low, high):
        self.low = vc(low)
        self.high = vc(high)
        torch_dist = torch.distributions.Uniform(low=self.low, high=self.high)
        super(Uniform, self).__init__(torch_dist)