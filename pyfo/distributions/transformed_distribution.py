#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  19:33
Date created:  26/01/2018

License: MIT
'''
import torch.nn as nn
from distributions.distribution_pyro import Distribution
import numbers
class TransformedDistribution(Distribution):
    """
    Transforms the base distribution by applying a sequence of `Bijector`s to it.
    This results in a scorable distribution (i.e. it has a `log_pdf()` method).
    :param base_distribution: a (continuous) base distribution; samples from this distribution
        are passed through the sequence of `Bijector`s to yield a sample from the
        `TransformedDistribution`
    :type base_distribution: pyro.distribution.Distribution
    :param bijectors: either a single Bijector or a sequence of Bijectors wrapped in a nn.ModuleList
    :returns: the transformed distribution
    """

    def __init__(self, base_distribution, bijectors, *args, **kwargs):
        super(TransformedDistribution, self).__init__(*args, **kwargs)
        self.reparameterized = base_distribution.reparameterized
        self.base_dist = base_distribution
        if isinstance(bijectors, Bijector):
            self.bijectors = nn.ModuleList([bijectors])
        elif isinstance(bijectors, nn.ModuleList):
            for bijector in bijectors:
                assert isinstance(bijector, Bijector), \
                    "bijectors must be a Bijector or a nn.ModuleList of Bijectors"
            self.bijectors = bijectors

    def sample(self, *args, **kwargs):
        """
        :returns: a sample y
        :rtype: torch.autograd.Variable
        Sample from base distribution and pass through bijector(s)
        """
        x = self.base_dist.sample(*args, **kwargs)
        next_input = x
        for bijector in self.bijectors:
            y = bijector(next_input)
            if bijector.add_inverse_to_cache:
                bijector._add_intermediate_to_cache(next_input, y, 'x')
            next_input = y
        return next_input

    def batch_shape(self, x=None, *args, **kwargs):
        return self.base_dist.batch_shape(x, *args, **kwargs)

    def event_shape(self, *args, **kwargs):
        return self.base_dist.event_shape(*args, **kwargs)

    def log_pdf(self, y, *args, **kwargs):
        """
        :param y: a value sampled from the transformed distribution
        :type y: torch.autograd.Variable
        :returns: the score (the log pdf) of y
        :rtype: torch.autograd.Variable
        Scores the sample by inverting the bijector(s) and computing the score using the score
        of the base distribution and the log det jacobian
        """
        value = y
        log_pdf = 0.0
        for bijector in reversed(self.bijectors):
            log_pdf -= bijector.log_det_jacobian(value, *args, **kwargs)
            value = bijector.inverse(value)
        log_pdf += self.base_dist.log_pdf(value, *args, **kwargs)
        return log_pdf

    def batch_log_pdf(self, y, *args, **kwargs):
        value = y
        log_det_jacobian = 0.0
        for bijector in reversed(self.bijectors):
            log_det_jacobian += bijector.batch_log_det_jacobian(value, *args, **kwargs)
            value = bijector.inverse(value)
        base_log_pdf = self.base_dist.batch_log_pdf(value, *args, **kwargs)
        if not isinstance(log_det_jacobian, numbers.Number):
            log_det_jacobian = log_det_jacobian.contiguous().view(*base_log_pdf.size())
            assert log_det_jacobian.size() == base_log_pdf.size(), \
                'Invalid batch_log_det_jacobian().size():\nexpected {}\nactual {}'.format(
                        base_log_pdf.size(), log_det_jacobian.size())
        return base_log_pdf - log_det_jacobian


class Bijector(nn.Module):
    """
    Abstract class `Bijector`. `Bijector` are bijective transformations with computable
    log det jacobians. They are meant for use in `TransformedDistribution`.
    """

    def __init__(self, *args, **kwargs):
        super(Bijector, self).__init__(*args, **kwargs)
        self.add_inverse_to_cache = False

    def __call__(self, *args, **kwargs):
        """
        Virtual forward method
        Invokes the bijection x=>y
        """
        raise NotImplementedError()

    def inverse(self, *args, **kwargs):
        """
        Virtual inverse method
        Inverts the bijection y => x.
        """
        raise NotImplementedError()

    def log_det_jacobian(self, *args, **kwargs):
        """
        Default logdet jacobian method.
        Computes the log det jacobian `|dy/dx|`
        """
        return self.batch_log_det_jacobian(*args, **kwargs).sum()

    def batch_log_det_jacobian(self, *args, **kwargs):
        """
        Virtual elementwise logdet jacobian method.
        Computes the log abs det jacobian `|dy/dx|`
        """
        raise NotImplementedError()