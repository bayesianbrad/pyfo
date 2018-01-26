import torch

from pyfo.distributions.Distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Gamma(TorchDistribution):
    r"""
    """
    def __init__(self, alpha, beta):
        self.alpha = vc(alpha)
        self.beta= vc(beta)
        torch_dist = torch.distributions.Gamma(concentration=self.alpha, rate=self.beta)
        super(Gamma, self).__init__(torch_dist)