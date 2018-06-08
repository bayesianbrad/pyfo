import torch

from distributions.distribution_wrapper import TorchDistribution
from utils.core import VariableCast as vc


class Normal(TorchDistribution):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    `loc` and `scale`.

    Example::

        >>> m = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mean of the distribution (often referred to as mu)
        scale (float or Tensor or Variable): standard deviation of the distribution
            (often referred to as sigma)2
    """
    def __init__(self, loc, scale):
        self.loc = vc(loc)
        self.scale = torch.sqrt(vc(scale))
        torch_dist = torch.distributions.Normal(loc=self.loc, scale=self.scale)
        super(Normal, self).__init__(torch_dist)