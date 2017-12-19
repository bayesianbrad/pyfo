#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  10:29
Date created:  13/11/2017

License: MIT
'''
from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyfo.distributions.distribution import Distribution
from pyfo.utils.core import VariableCast


class MultivariateNormal(Distribution):
    """
    Univariate normal (Gaussian) distribution.

    A distribution over tensors in which each element is independent and
    Gaussian distributed, with its own mean and standard deviation. The
    distribution is over tensors that have the same shape as the parameters `mu`
    and `sigma`, which in turn must have the same shape as each other.

    This is often used in conjunction with `torch.nn.Softplus` to ensure the
    `sigma` parameters are positive.

    :param torch.autograd.Variable mu: Means.
    :param torch.autograd.Variable sigma: Standard deviations.
        Should be positive and the same shape as `mu`.
    """
    reparameterized = True

    def __init__(self, mu, cov, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        self.mu = VariableCast(mu)
        self.cov = VariableCast(cov)
        self.log_pdf_mask = log_pdf_mask
        assert self.mean.data.size()[0] == self.cov.size()[0]  # , "ERROR! mean and cov have different size!")
        self.chol_std = torch.t(torch.potrf(self.cov)) # lower triangle
        self.chol_std_inv = torch.inverse(self.chol_std) # cholesky decompositon
        if self.mu.size()[1] != self.cov.size()[1]:
            raise ValueError("Expected mu.size()[1] == sigma.size()[1], but got {} vs {}".format(mu.size()[1], cov.size()[1]))
        if self.mu.dim() == 1 and batch_size is not None:
            self.mu = self.mu.expand(batch_size, mu.size(0))
            self.sigma = self.cov.expand(batch_size, cov.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(MultivariateNormal, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        mu = self.mu
        if x is not None:
            if x.size()[-event_dim] != mu.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.mu.size()[-1], but got {} vs {}".format(
                                     x.size(-1), mu.size(-1)))
            try:
                mu = self.mu.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `mu` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(mu.size(), x.size(), str(e)))
        return mu.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.mu.size()[-event_dim:]

    def sample(self):
        """
        Reparameterized Normal sampler.

        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        eps = Variable(torch.randn(self.mu.size()).type_as(self.mu.data))
        z = self.mu + eps * self.sigma
        return z if self.reparameterized else z.detach()

    def batch_log_pdf(self, x):
        """
        Diagonal Normal log-likelihood

        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        # expand to patch size of input
        x = VariableCast(x)
        mu = self.mu.expand(self.shape(x))
        sigma = self.sigma.expand(self.shape(x))
        log_pxs = -1 * (torch.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * torch.pow((x - mu) / sigma, 2))
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_pxs = log_pxs * self.log_pdf_mask
        batch_log_pdf = torch.sum(log_pxs, -1)
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        return batch_log_pdf.contiguous().view(batch_log_pdf_shape)

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`
        """
        return self.mu

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`
        """
        return torch.pow(self.sigma, 2)

    def is_discrete(self):
        """
            Ref: :py:meth:`pyro.distributions.distribution.Distribution.is_discrete`.
        """
        return False



    #Our MVN
    # class MultivariateNormal():
    #     """Normal random variable"""
    #
    #     def __init__(self, mean, cov):
    #         """Initialize this distribution with mean, cov.
    #         input:
    #             mean: n by 1
    #             cov: covariance matrix, n by n
    #         """
    #         self.mean = VariableCast(mean)
    #         self.cov = VariableCast(cov)
    #         assert self.mean.data.size()[0] == self.cov.data.size()[0]  # , "ERROR! mean and cov have different size!")
    #         self.dim = self.mean.data.size()[0]
    #         self.chol_std = VariableCast(torch.t(torch.potrf(self.cov.data)))  # lower triangle
    #         self.chol_std_inv = torch.inverse(self.chol_std)
    #
    #     def sample(self, num_samples=1):
    #         zs = torch.randn(self.dim, 1)
    #         # print("zs", zs)
    #         samples = Variable(self.mean.data + torch.matmul(self.chol_std.data, zs), requires_grad=True)
    #         return samples
    #
    #     def logpdf(self, value):
    #         """
    #         value : obs value, should be n by 1
    #         :return: scalar, log pdf value
    #         """
    #         value = VariableCast(value)
    #         cov_det = self.chol_std.diag().prod() ** 2
    #         log_norm_constant = 0.5 * self.dim * torch.log(torch.Tensor([2 * np.pi])) \
    #                             + 0.5 * torch.log(cov_det.data)
    #         right = torch.matmul(self.chol_std_inv, value - self.mean)
    #         # print(value, self.mean, value - self.mean)
    #         log_p = - Variable(log_norm_constant) - 0.5 * torch.matmul(torch.t(right), right)
    #         return log_p