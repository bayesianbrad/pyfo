import torch

from pyfo.distributions.Distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class FisherSnedecor(TorchDistribution):
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
    def __init__(self, df1, df2):
        self.df1 = vc(df1)
        self.df2 = vc(df2)
        torch_dist = torch.distributions.FisherSnedecor(df1=self.df1, df2=self.df2)
        super(FisherSnedecor, self).__init__(torch_dist)