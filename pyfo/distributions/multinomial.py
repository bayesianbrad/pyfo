import torch

from ..distributions.distribution_wrapper import TorchDistribution
from ..utils.core import VariableCast as vc


class Multinomial(TorchDistribution):
    r"""
    Creates a Fisher-Snedecor distribution parameterized by `df1` and `df2`.

    Creates a Multinomial distribution parameterized by `total_count` and
    either `probs` or `logits` (but not both). The innermost dimension of
    `probs` indexes over categories. All other dimensions index over batches.

    Note that `total_count` need not be specified if only :meth:`log_prob` is
    called (see example below)

    -   :meth:`sample` requires a single shared `total_count` for all
        parameters and samples.
    -   :meth:`log_prob` allows different `total_count` for each parameter and
        sample.

    Example::

        >>> m = Multinomial(100, torch.Tensor([ 1, 1, 1, 1]))
        >>> x = m.sample()  # equal probability of 0, 1, 2, 3
         21
         24
         30
         25
        [torch.FloatTensor of size 4]]

        >>> Multinomial(probs=torch.Tensor([1, 1, 1, 1])).log_prob(x)
        -4.1338
        [torch.FloatTensor of size 1]
    Args:
        df1 (float or Tensor or Variable): degrees of freedom parameter 1
        df2 (float or Tensor or Variable): degrees of freedom parameter 2
    """
    def __init__(self, total_count=1, probs=None, logits=None):
        self.probs = vc(probs)
        self.logits = vc(logits)
        torch_dist = torch.distributions.Multinomial(total_count=total_count,probs=self.probs, logits=self.logits)
        super(Multinomial, self).__init__(torch_dist)