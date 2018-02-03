import torch

from pyfo.distributions.distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Pareto(TorchDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> m = Pareto(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
         1.5623
        [torch.FloatTensor of size 1]

    Args:
        scale (float or Tensor or Variable): Scale parameter of the distribution
        alpha (float or Tensor or Variable): Shape parameter of the distribution
    """
    def __init__(self, scale, alpha):
        self.scale = vc(scale)
        self.alpha = vc(alpha)
        torch_dist = torch.distributions.Pareto(scale=self.scale, alpha=self.alpha)
        super(Pareto, self).__init__(torch_dist)