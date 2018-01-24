import torch
from torch.autograd import Variable

from pyfo.utils.core import VariableCast


class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by `total_count` and
    either `probs` or `logits` (but not both).

    -   Requires a single shared `total_count` for all
        parameters and samples.

    Example::

        >>> m = Binomial(100, torch.Tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
         0
         22
         71
         100
        [torch.FloatTensor of size 4]]

    Args:
        total_count (int): number of Bernoulli trials
        probs (Tensor or Variable): Event probabilities
        logits (Tensor or Variable): Event log-odds
    """

    def __init__(self, total_count=1, probs=None, logits=None):
        self.total_count = total_count
        self.probs = VariableCast(probs)
        self.logits = VariableCast(logits)
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        self._param = self.probs if probs is not None else self.logits
        torch_dist = torch.distributions.Binomial(total_count=self.total_count, probs=self._param, logits=self.logits)
        super(Binomial, self).__init__(torch_dist)
