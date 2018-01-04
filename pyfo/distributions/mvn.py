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


def potri_compat(var):
    return Variable(torch.potri(var.data)) if torch.__version__ < '0.3.0' else torch.potri(var)


def potrf_compat(var):
    return Variable(torch.potrf(var.data)) if torch.__version__ < '0.3.0' else torch.potrf(var)


def linear_solve_compat(matrix, matrix_chol, y):
    """Solves the equation ``torch.mm(matrix, x) = y`` for x."""
    assert matrix.requires_grad == matrix_chol.requires_grad
    if matrix.requires_grad or y.requires_grad:
        # If derivatives are required, use the more expensive gesv.
        return torch.gesv(y, matrix)[0]
    else:
        # Use the cheaper Cholesky solver.
        return torch.potrs(y, matrix_chol)


def matrix_inverse_compat(matrix, matrix_chol):
    """Computes the inverse of a positive semidefinite square matrix."""
    assert matrix.requires_grad == matrix_chol.requires_grad
    if matrix.requires_grad:
        # If derivatives are required, use the more expensive inverse.
        return torch.inverse(matrix)
    else:
        # Use the cheaper Cholesky based potri.
        return potri_compat(matrix_chol)


class MultivariateNormal(Distribution):
    """Multivariate normal (Gaussian) distribution.
    A distribution over vectors in which all the elements have a joint Gaussian
    density.
    :param torch.autograd.Variable mu: Mean. Must be a vector (Variable
        containing a 1d Tensor).
    :param torch.autograd.Variable covariance_matrix: Covariance matrix.
        Must be symmetric and positive semidefinite.
    :param torch.autograd.Variable scale_tril: The Cholesky decomposition of
        the covariance matrix. You can pass this instead of `covariance_matrix`.
    :param use_inverse_for_batch_log: If this is set to true, the torch.inverse
        function will be used to compute log_pdf. This means that the results of
        log_pdf can be differentiated with respect to the covariance matrix.
        Since the gradient of torch.potri is currently not implemented,
        differentiation of log_pdf wrt. covariance matrix is not possible when
        using the Cholesky decomposition. Using the Cholesky decomposition is
        however much faster and therefore enabled by default.
    :param normalized: If set to `False` the normalization constant is omitted
        in the results of batch_log_pdf and log_pdf. This might be preferable,
        as computing the determinant of the covariance matrix might not always
        be numerically stable. Defaults to `True`.
    :raises: ValueError if the shape of any parameter is not supported.
    """

    reparameterized = True

    def __init__(self, mu, covariance_matrix=None, scale_tril=None, batch_size=None, use_inverse_for_batch_log=False,
                 normalized=True, *args, **kwargs):
        if covariance_matrix is None and scale_tril is None:
            raise ValueError('At least one of covariance_matrix or scale_tril must be specified')
        if covariance_matrix is not None:
            cov_matrix  = VariableCast(covariance_matrix)
        if scale_tril is not None:
            scale_tri = VariableCast(scale_tril)
        self.mu = VariableCast(mu)
        self.output_shape = self.mu.size()
        self.use_inverse_for_batch_log = use_inverse_for_batch_log
        self.normalized = normalized
        self.batch_size = batch_size if batch_size is not None else 1
        if scale_tril is not None and covariance_matrix is not None:
            raise ValueError("Only either 'covariance_matrix' or 'scale_tril' can be passed at once.")
        if cov_matrix is not None:
            if not cov_matrix.dim() == 2:
                raise ValueError("The covariance matrix must be a matrix, but got covariance_matrix.size() = {}".format(
                    self.mu.size()))
            self.sigma = VariableCast(cov_matrix)
            self.sigma_cholesky = potrf_compat(cov_matrix)
        else:
            if not scale_tri.dim() == 2:
                raise ValueError("The Cholesky decomposition of the covariance matrix must be a matrix, "
                                 "but got scale_tril.size() = {}".format(
                                     self.mu.size()))
            self.sigma = torch.mm(scale_tri.transpose(0, 1), scale_tri)
            self.sigma_cholesky = scale_tri
        if self.mu.dim() > 1:
            raise ValueError("The mean must be a vector, but got mu.size() = {}".format(self.mu.size()))

        super(MultivariateNormal, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        mu = self.mu.expand(self.batch_size, *self.mu.size()).squeeze(0)
        X = VariableCast(x)
        if X is not None:
            if X.size()[-1] != mu.size()[-1]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.mu.size()[0], but got {} vs {}".format(
                                     X.size(-1), mu.size(-1)))
            try:
                mu = mu.expand_as(X)
            except RuntimeError as e:
                raise ValueError("Parameter `mu` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(mu.size(), x.size(), str(e)))

        return mu.size()[:-1]

    def event_shape(self):
        return self.mu.size()[-1:]

    def sample(self):
        """Generate a sample with the specified covariance matrix and mean.
        Differentiation wrt. to the covariance matrix is only supported on
        PyTorch version 0.3.0 or higher.
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`
        """
        batch_size = self.batch_size
        uncorrelated_standard_sample = Variable(torch.randn(
            batch_size, *self.mu.size()).type_as(self.mu.data))
        transformed_sample = self.mu + \
            torch.mm(uncorrelated_standard_sample, self.sigma_cholesky)
        return transformed_sample if self.reparameterized else transformed_sample.detach()

    def batch_log_pdf(self, x):
        if not self.normalized and self.sigma_cholesky.requires_grad:
            raise NotImplementedError("Differentiation is not supported (yet) if normalized=False.")
        X =VariableCast(x)

        batch_size = X.size()[0] if len(X.size()) > len(self.mu.size()) else 1
        batch_log_pdf_shape = self.batch_shape(X) + (1,)
        X = X.view(batch_size, *self.mu.size())
        # TODO Implement the gradient of the normalization factor if normalized=False
        normalization_factor = torch.log(
            self.sigma_cholesky.diag()).sum() + (self.mu.size()[0] / 2) * np.log(2 * np.pi) if self.normalized else 0
        # TODO It may be useful to switch between matrix_inverse_compat() and linear_solve_compat() based on the
        # batch size and the size of the covariance matrix
        sigma_inverse = matrix_inverse_compat(self.sigma, self.sigma_cholesky)
        return -(normalization_factor + 0.5 * torch.sum((X - self.mu).unsqueeze(2) * torch.bmm(
            sigma_inverse.expand(batch_size, *self.sigma_cholesky.size()),
            (X - self.mu).unsqueeze(-1)), 1)).view(batch_log_pdf_shape)

    def analytic_mean(self):
        return self.mu

    def analytic_var(self):
        return torch.diag(self.sigma)    #         return log_p