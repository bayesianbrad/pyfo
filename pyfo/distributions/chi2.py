import torch

from distributions.distribution_wrapper import TorchDistribution
from utils.core import VariableCast as vc


class Chi2(TorchDistribution):
    r"""
    Creates a Chi2 distribution parameterized by shape parameter `df`.
    This is exactly equivalent to Gamma(alpha=0.5*df, beta=0.5)

    Example::

        >>> m = Chi2(torch.Tensor([1.0]))
        >>> m.sample()  # Chi2 distributed with shape df=1
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        df (float or Tensor or Variable): shape parameter of the distribution
    """

    def __init__(self, df):
        self.df = vc(df)
        torch_dist = torch.distributions.Chi2(df=self.df)
        super(Chi2, self).__init__(torch_dist)
