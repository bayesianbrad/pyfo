import torch

from pyfo.distributions.Distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Laplace(TorchDistribution):
    r"""
    Creates a Laplace distribution parameterized by `loc` and 'scale'.

    Example::

        >>> m = Laplace(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mean of the distribution
        scale (float or Tensor or Variable): scale of the distribution
    """
    def __init__(self, loc, scale):
        self.loc = vc(loc)
        self.scale = vc(scale)
        torch_dist = torch.distributions.Laplace(loc=self.loc, scale=self.scale)
        super(Laplace, self).__init__(torch_dist)