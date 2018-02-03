import torch

from pyfo.distributions.distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class OneHotCategorical(TorchDistribution):
    r"""
    Creates a one-hot categorical distribution parameterized by `probs`.

    Samples are one-hot coded vectors of size probs.size(-1).

    See also: :func:`torch.distributions.Categorical`

    Example::

        >>> m = OneHotCategorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
         0
         0
         1
         0
        [torch.FloatTensor of size 4]

    Args:
        probs (Tensor or Variable): event probabilities
    """
    def __init__(self, probs, logits):
        self.probs = vc(probs)
        self.logits = vc(logits)
        torch_dist = torch.distributions.OneHotCategorical(probs=self.probs, logits=self.logits)
        super(OneHotCategorical, self).__init__(torch_dist)