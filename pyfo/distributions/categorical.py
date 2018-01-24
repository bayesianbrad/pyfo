import torch

from pyfo.distributions.Distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class Categorical(TorchDistribution):
    r"""
    Creates a categorical distribution parameterized by either `probs` or
    `logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from `0 ... K-1` where `K` is probs.size(-1).

    If `probs` is 1D with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is 2D, it is treated as a batch of probability vectors.

    See also: :func:`torch.multinomial`

    Example::

        >>> m = Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         3
        [torch.LongTensor of size 1]

    Args:
        probs (Tensor or Variable): event probabilities
        logits (Tensor or Variable): event log probabilities
    """
    def __init__(self, probs=None, logits=None):
        self.probs = vc(probs)
        self.logits = vc(logits)
        self._param = self.probs if probs is not None else self.logits
        torch_dist = torch.distributions.Categorical(probs=self._param, logits=self.logits)
        super(Categorical, self).__init__(torch_dist)


