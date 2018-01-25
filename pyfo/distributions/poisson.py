from __future__ import absolute_import, division, print_function

import numpy.random as npr
from torch.autograd import Variable
from pyfo.distributions.distribution_pyro import Distribution
from pyfo.utils.core import VariableCast


class Poisson(Distribution):
    """
    Poisson distribution over integers parameterized by scale `lambda`.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `lam` parameter is positive.

    :param torch.autograd.Variable lam: Mean parameter (a.k.a. `lambda`).
        Should be positive.
    """

    def __init__(self, lam, batch_size=None, *args, **kwargs):
        self.lam = VariableCast(lam)
        if self.lam.dim() == 1 and batch_size is not None:
            self.lam = self.lam.expand(batch_size, self.lam.size(0))
        super(Poisson, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        lam = self.lam
        if x is not None:
            if x.size()[-event_dim] != lam.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.lam.size()[-1], but got {} vs {}".format(
                                     x.size(-1), lam.size(-1)))
            try:
                lam = self.lam.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `lam` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(lam.size(), x.size(), str(e)))
        return lam.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.lam.size()[-event_dim:]

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        x = npr.poisson(lam=self.lam.data.cpu().numpy()).astype("float")
        return Variable(torch.Tensor(x).type_as(self.lam.data))

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        x = VariableCast(x)
        lam = self.lam.expand(self.shape(x))
        ll_1 = torch.sum(x * torch.log(lam), -1)
        ll_2 = -torch.sum(lam, -1)
        ll_3 = -torch.sum(torch.lgamma(x + 1.0), -1)
        batch_log_pdf = ll_1 + ll_2 + ll_3
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.lam

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return self.lam

    def is_discrete(self):
        """
            Ref: :py:meth:`pyro.distributions.distribution.Distribution.is_discrete`.
        """
        return True

import torch


# class Poisson(TorchDistribution):
#     r"""
#     Samples from a Pareto Type 1 distribution.
#
#     Example::
#
#         >>> m = Pareto(torch.Tensor([1.0]), torch.Tensor([1.0]))
#         >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
#          1.5623
#         [torch.FloatTensor of size 1]
#
#     Args:
#         scale (float or Tensor or Variable): Scale parameter of the distribution
#         alpha (float or Tensor or Variable): Shape parameter of the distribution
#     """
#     def __init__(self, lam):
#         self.lam = vc(lam)
#         torch_dist = torch.distributions.Poisson(lam=self.lam)
#         super(Poisson, self).__init__(torch_dist)