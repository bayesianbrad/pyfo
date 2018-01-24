import torch

from pyfo.distributions.Distribution_wrapper import TorchDistribution
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
    def __init__(self, rate):
        self.rate =vc(rate)
        torch_dist = torch.distributions.Dirichlet(rate=self.rate)
        super(Exponential, self).__init__(torch_dist)