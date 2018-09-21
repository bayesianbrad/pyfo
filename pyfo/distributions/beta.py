import torch

from ..distributions.distribution_wrapper import TorchDistribution
from ..utils.core import VariableCast


class Beta(TorchDistribution):
    r"""
    Beta distribution parameterized by `concentration1` and `concentration0`.

    Example::

        >>> m = Beta(torch.Tensor([0.5]), torch.Tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        concentration1 (float or Tensor or Variable): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor or Variable): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    def __init__(self, alpha, beta):
        alpha = VariableCast(alpha)
        beta = VariableCast(beta)
        torch_dist = torch.distributions.Beta(concentration0=alpha,concentration1=beta)
        super(Beta, self).__init__(torch_dist)