import torch

from ..distributions.distribution_wrapper import TorchDistribution
from ..utils.core import VariableCast as vc


class Gumbel(TorchDistribution):
    r"""
    Creates a Fisher-Snedecor distribution parameterized by `df1` and `df2`.

    Example::

        >>> m = FisherSnedecor(torch.Tensor([1.0]), torch.Tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
         0.2453
        [torch.FloatTensor of size 1]

    Args:
        df1 (float or Tensor or Variable): degrees of freedom parameter 1
        df2 (float or Tensor or Variable): degrees of freedom parameter 2
    """
    def __init__(self, loc, scale):
        self.loc = vc(loc)
        self.scale = vc(scale)
        torch_dist = torch.distributions.Gumbel(loc=self.loc, scale=self.scale)
        super(Gumbel, self).__init__(torch_dist)