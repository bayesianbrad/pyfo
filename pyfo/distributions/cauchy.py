import torch

from ..distributions.distribution_wrapper import TorchDistribution
from ..utils.core import VariableCast as vc


class Cauchy(TorchDistribution):
    r"""
    Samples from a Cauchy (Lorentz) distribution. The distribution of the ratio of
    independent normally distributed random variables with means `0` follows a
    Cauchy distribution.

    Example::

        >>> m = Cauchy(torch.Tensor([0.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Cauchy distribution with loc=0 and scale=1
         2.3214
        [torch.FloatTensor of size 1]

    Args:
        loc (float or Tensor or Variable): mode or median of the distribution.
        scale (float or Tensor or Variable): half width at half maximum.
    """

    def __init__(self, loc, scale):
        self.loc, self.scale = vc(loc), vc(scale)
        torch_dist = torch.distributions.Cauchy(loc=self.loc, scale=self.scale)
        super(Cauchy, self).__init__(torch_dist)
