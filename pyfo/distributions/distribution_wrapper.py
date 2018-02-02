#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:28
Date created:  24/01/2018

License: MIT
'''


import torch
import torch.distributions.constraints as constraints
from torch.distributions.constraint_registry import biject_to
import torch.distributions.transforms as transforms
from torch.distributions import TransformedDistribution as td
from pyfo.utils.core import VariableCast

class TorchDistribution():
    """
    A very thin wrapper around torch distributions
    """

    def __init__(self, torch_dist, transformed=False, name=None):
        super(TorchDistribution, self).__init__()
        self.torch_dist = torch_dist
        self._sample_shape = torch.Size()
        self._transformed = transformed
        self._name  = name
        if self._transformed:
            self.transform_dist = td(self.torch_dist, [transforms.ExpTransform(0)])

    def sample(self):
        if self._transformed:
            """
            Only care about distribution variables for now, not the support
            of the parameters.
            """
            if self._name =='Gamma':
                return  self.transformed_dist.sample()
            if self._name == 'Exponential':
                return  self.transform_dist.sample()

        return self.torch_dist.sample(self._sample_shape).type(torch.FloatTensor)

    def log_pdf(self, x):
        if self._transformed:
            """
            Transform x back to the right support and then evaluates log_pdf
            + jjacobian factor
            if self._name == 'Gamma':
                do .....
            return self.torch_dist.log_prob(x_constrained) + |J|
            if self._name == 'Exponential'
                do .....

            """
            x = VariableCast(x)
            if self._name =='Gamma':
               return -self.transform_dist.sample()
            if self._name == 'Exponential':
                return -self.transform_dist.log_prob(x)
        else:
            x = VariableCast(x)
            return self.torch_dist.log_prob(x)