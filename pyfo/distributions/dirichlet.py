import torch

from ..distributions.distribution_wrapper import TorchDistribution
from ..utils.core import VariableCast as vc


class Dirichlet(TorchDistribution):
    r"""
    Creates a Dirichlet distribution parameterized by concentration `concentration`.

    Example::

        >>> m = Dirichlet(torch.Tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentrarion concentration
         0.1046
         0.8954
        [torch.FloatTensor of size 2]

    Args:
        concentration (Tensor or Variable): concentration parameter of the distribution
            (often referred to as alpha)
    """

    def __init__(self, alpha):
        self.alpha = vc(alpha)
        torch_dist = torch.distributions.Dirichlet(concentration=self.alpha)
        super(Dirichlet, self).__init__(torch_dist)