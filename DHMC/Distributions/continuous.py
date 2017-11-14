#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:43
Date created:  09/11/2017

License: MIT
'''
import numpy as np
import torch as torch
from DHMC.utils.core import VariableCast

class ContinuousRandomVariable():
    '''A very basic representation of what should be contained in a
       continuous random variable class'''
    def pdf(self):
        raise NotImplementedError("pdf is not implemented")
    def logpdf(self, x):
        raise NotImplementedError("logpdf is not implemented")

    def sample(self):
        raise NotImplementedError("sample is not implemented")
    def isdiscrete(self):
        return False

class Normal(ContinuousRandomVariable):
    """1d Normal random variable"""
    def __init__(self, mean, std):
        """Initialize this distribution with mean, variance.

        input:
            mean:
            std:  standard deviation
        """

        self.mean = VariableCast(mean)
        if len(mean.data.shape) == 1:
            self.mean = mean.unsqueeze(1)
        self.std = VariableCast(std)


    def sample(self, num_samples = 1):
        x = torch.randn(1)
        sample = Variable(x * self.std.data + self.mean.data, requires_grad = True)
        return sample #.detach()


    def logpdf(self, value):
        mean = self.mean
        var = self.std ** 2
        value = VariableCast(value)
        if len(value.data.shape) == 1:
            value = value.unsqueeze(1)
        if isinstance(mean, torch.IntTensor):
            mean = mean.type(torch.FloatTensor)
        # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
        return (-0.5 * torch.pow(value - mean, 2) / var - 0.5 * torch.log(2 * var * np.pi))


class MultivariateNormal(ContinuousRandomVariable):
    """Normal random variable"""
    def __init__(self, mean, cov):
        """Initialize this distribution with mean, cov.

        input:
            mean: n by 1
            cov: covariance matrix, n by n
        """
        self.mean = VariableCast(mean)
        self.cov = VariableCast(cov)
        assert self.mean.data.size()[0] == self.cov.data.size()[0] #, "ERROR! mean and cov have different size!")
        self.dim = self.mean.data.size()[0]
        self.chol_std = VariableCast(torch.t(torch.potrf(self.cov.data)))  # lower triangle
        self.chol_std_inv = torch.inverse(self.chol_std)

    def sample(self, num_samples=1 ):
        zs = torch.randn(self.dim, 1)
        # print("zs", zs)
        samples = Variable( self.mean.data + torch.matmul(self.chol_std.data, zs), requires_grad = True)
        return samples

    def logpdf(self, value):
        """
        value : obs value, should be n by 1
        :return: scalar, log pdf value
        """
        value = VariableCast(value)
        cov_det = self.chol_std.diag().prod() ** 2
        log_norm_constant = 0.5 * self.dim * torch.log(torch.Tensor([2 * np.pi])) \
                            + 0.5*torch.log(cov_det.data)
        right = torch.matmul( self.chol_std_inv, value - self.mean)
        # print(value, self.mean, value - self.mean)
        log_p = - Variable(log_norm_constant) - 0.5 * torch.matmul(torch.t(right), right)
        return log_p

# class Laplace(ContinuousRandomVariable):
#     '''
#     Laplace random variable
#
#     Methods
#     -------
#     sample X ~ Laplace(location, scale)
#     logpdf
#
#     Attributes
#     ----------
#     location - Type torch.autograd.Variable, torch.Tensor, nparray
#                Size \mathbb{R}^{1 x N}
#     scale    - Type torch.autograd.Variable, torch.Tensor, nparray
#                Size \mathbb{R}^{1 x N}
#     '''
#     def __init__(self, location, scale):
#         self.location = VariableCast(location)
#         self.scale    = VariableCast(scale)
#     def sample(self):
#         # https: // en.wikipedia.org / wiki / Laplace_distribution
#         uniforms = torch.Tensor(self.location.size()).uniform_() - 0.5
#         uniforms = VariableCast(uniforms)
#         return self.location - self._scale * torch.sign(uniforms) * \
#                                 torch.log(1 - 2 * torch.abs(uniforms))
#
#     def logpdf(self, value):
#         return -torch.div(torch.abs(value - self._location),self._scale) - torch.log(2 * self._scale)
#
# class Normal(ContinuousRandomVariable):
#     """Normal random variable
#     Returns  a normal distribution object class with mean - mean
#     and standard deviation - std.
#
#     Methods
#     --------
#     sample   - Returns a sample X ~ N(mean,std) as a Variable.
#     pdf
#     logpdf   -
#
#     Attributes
#     ----------
#     mean     - Type: torch.autograd.Variable, torch.Tensor, nparray
#                Size: \mathbb{R}^{1 x N}
#     std      - Type: torch.autograd.Variable, torch.Tensor, nparray
#                Size: \mathbb{R}^{1 x N}
#
#     """
#     def __init__(self, mean, std):
#         self.mean = VariableCast(mean)
#         self.std  = VariableCast(std)
#
#
#     def sample(self, num_samples = 1):
#         # x ~ N(0,1)
#         self.unifomNorm = Variable(torch.randn(1,num_samples))
#         return self.unifomNorm * (self.std) + self.mean
#
#     def logpdf(self, value):
#         value = VariableCast(value)
#         # pdf: 1 / torch.sqrt(2 * var * np.pi) * torch.exp(-0.5 * torch.pow(value - mean, 2) / var)
#         return (-0.5 *  torch.pow(value - self.mean, 2) / self.std**2) -  torch.log(self.std)
#
# class MultivariateNormal(ContinuousRandomVariable):
#    """Normal random variable"""
#    def __init__(self, mean, cov):
#        """Initialize this distribution with mean, cov.
#
#        input:
#            mean: n by 1
#            cov: covariance matrix, n by n
#        """
#        self.mean = VariableCast(mean)
#        self.cov = VariableCast(cov)
#        assert self.mean.data.size()[0] == self.cov.data.size()[0] #, "ERROR! mean and cov have different size!")
#        self.dim = self.mean.data.size()[0]
#        self.chol_std = VariableCast(torch.potrf(self.cov.data).t())  # lower triangle
#        self.chol_std_inv = torch.inverse(self.chol_std)
#
#    def sample(self, num_samples=1):
#        zs = torch.randn(self.dim, 1)
#        # print("zs", zs)
#        # samples = Variable( self.mean.data + torch.matmul(self.chol_std.data, zs), requires_grad = True)
#        return self.mean.data + torch.matmul(self.chol_std.data, zs)
#
#    def logpdf(self, value):
#        """
#        value : obs value, should be n by 1
#        :return: scalar, log pdf value
#        """
#        value = VariableCast(value)
#        cov_det = self.chol_std.diag().prod() ** 2
#        log_norm_constant = 0.5 * self.dim * torch.log(torch.Tensor([2 * np.pi])) \
#                            + 0.5*torch.log(cov_det.data)
#        right = torch.matmul( self.chol_std_inv, value - self.mean)
#        # print(value, self.mean, value - self.mean)
#        log_p = - Variable(log_norm_constant) - 0.5 * torch.matmul(torch.t(right), right)
#        return log_p

# class MultivariateNormal(ContinuousRandomVariable):
#     """MultivariateIndependentNormal simple class
#     Returns  a normal distribution object class with mean - mean
#     and standard deviation - std.
#
#     Methods
#     --------
#     sample   - Returns a sample X ~ N(mean,std) as a Variable. Takes an additional
#                argument grad. If we need the differential of X ~ N(mu, var)
#     pdf
#     logpdf   -
#
#     Attributes
#     ----------
#     mean        -  Type: torch.Tensor, Variable, ndarray
#                    Size: [ 1, ...., N]
#     covariance  -  Type: torch.Tensor, Variable, ndarray
#                     Size: \mathbb{R}^{N \times N}
#     """
#
#     def __init__(self, mean, covariance):
#         """Initialize this distribution with given mean and covariance.
#         """
#         assert (mean.size()[0] == covariance.size()[0])
#         assert (mean.size()[0] == covariance.size()[1])
#         self.mean      = VariableCast(mean)
#         self.covariance = VariableCast(covariance)
#         # cholesky decomposition returns upper triangular matrix. Will not accept Variables
#         self.L = VariableCast(torch.potrf(self.covariance.data))
#
#     def sample(self):
#         # Returns a sample of a multivariate normal X ~ N(mean, cov)
#         # A column vecotor of X ~ N(0,I)  - IS THIS ACTUALLY TRUE ?? NOT SURE, ALL POINTS HAVE
#         # THE RIGHT MEAN AND VARIANCE N(0,1)
#         self.uniformNorm  = torch.Tensor(self.mean.size()).normal_()
#         samples           = self.mean.data  + Variable(self.L.t().data.mm(self.uniformNorm))
#         return samples
#
#     # def sample_grad(self):
#     #     x          = Variable(self.uniformNorm.data, requires_grad = True)
#     #     logSample  = self.uniformNorm * (self.L)
#     #     sampleGrad = torch.autograd.grad([logSample],[x], grad_outputs= torch.ones(x.size()))[0]
#     #     return sampleGrad
#     # def pdf(self, value):
#     #     assert (value.size() == self._mean.size())
#     #     # CAUTION: If the covariance is 'Unknown' then we will
#     #     # not be returned the correct derivatives.
#     #     print('****** Warning ******')
#     #     print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
#     #     WILL NOT RECORD THE GRAPH STRUCTURE OF THE FULL PASS, ONLY THE CALCUALTION OF THE PDF')
#     #     value = VariableCast(value)
#     #     # the sqrt root of a det(cov) : sqrt(det(cov)) == det(L.t()) = \Pi_{i=0}^{N} L_{ii}
#     #     self._constant = torch.pow(2 * np.pi, value.size()[1]) * self._L.t().diag().prod()
#     #     return self._constant * torch.exp(
#     #         -0.5 * (value - self._mean).mm(self._L.inverse().mm(self._L.inverse().t())).mm(
#     #             (value - self._mean).t()))
#
#     def logpdf(self, value):
#         print('****** Warning ******')
#         print(' IF COVARIANCE IS UNKNOWN AND THE DERIVATIVES ARE NEEDED W.R.T IT, THIS RETURNED FUNCTION \n \
#         WILL NOT RECORD THE GRAPH STRUCTURE OF THE FULL PASS, ONLY THE CALCUALTION OF THE LOGPDF')
#         assert (value.size() == self.mean.size())
#         value = VariableCast(value)
#         self._constant = ((2 * np.pi) ** value.size()[1]) * self.L.t().diag().prod()
#         return torch.log(-0.5 * (value - self.mean).t().mm(self.L.inverse().mm(self.L.inverse().t())).mm(
#             (value - self.mean))) \
#                + self._constant
#
