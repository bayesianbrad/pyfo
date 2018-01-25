import torch

from pyfo.distributions.Distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Bernoulli(TorchDistribution):
    r"""
    Creates a Bernoulli distribution parameterized by `probs` or `logits`.

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> m = Bernoulli(0.3)
        >>> m.sample()  # 30% chance 1; 70% chance 0
         0.0
        [torch.FloatTensor of size 1]

    Args:
        probs (Number, Tensor or Variable): the probabilty of sampling `1`
        logits (Number, Tensor or Variable): the log-odds of sampling `1`
    """

    def __init__(self, probs=None, logits=None):
        self.prob = vc(probs)
        self.logits = vc(logits)
        torch_dist = torch.distributions.Bernoulli(probs=self.probs, logits=self.logits)
        super(Bernoulli, self).__init__(torch_dist)


