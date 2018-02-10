import torch

from pyfo.distributions.distribution_wrapper import TorchDistribution
from pyfo.utils.core import VariableCast as vc


class StudentT(TorchDistribution):
    r"""
    Creates a Student's t-distribution parameterized by `df`.

    Example::

        >>> m = StudentT(torch.Tensor([2.0]))
        >>> m.sample()  # Student's t-distributed with degrees of freedom=2
         0.1046
        [torch.FloatTensor of size 1]

    Args:
        df (float or Tensor or Variable): degrees of freedom
    """
    def __init__(self, df, loc,scale):
        self.scale = vc(scale)
        self.loc = vc(loc)
        self.df = vc(df)
        torch_dist = torch.distributions.StudentT(df=self.df, loc=self.loc, scale=self.scale)
        super(StudentT, self).__init__(torch_dist)