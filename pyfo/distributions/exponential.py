import torch

from pyfo.distributions.distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Exponential(TorchDistribution):
    r"""
    Creates a Exponential distribution parameterized by `rate`.

    Example::

        >>> m = Exponential(torch.Tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        rate (float or Tensor or Variable): rate = 1 / scale of the distribution
    """
    def __init__(self, rate, transformed=True, name='Exponential'):
        self.rate =vc(rate)
        self.name ='Exponential'
        self._transformed = transformed
        torch_dist = torch.distributions.Exponential(rate=self.rate)
        super(Exponential, self).__init__(torch_dist, transformed=self._transformed, name=self.name )