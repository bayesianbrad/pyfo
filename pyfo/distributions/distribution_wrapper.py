#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:28
Date created:  24/01/2018

License: MIT
'''


import torch
from pyfo.utils.core import VariableCast

class TorchDistribution():
    """
    A very thin wrapper around torch distributions
    """

    def __init__(self, torch_dist, Transformed=False, name=None):
        super(TorchDistribution, self).__init__()
        self.torch_dist = torch_dist
        self._sample_shape = torch.Size()
        self._transformed = Transformed
        self._name  = name

    def sample(self, sample_size=None):
        # if self._transformed:
        #     """
        #     Transform sample back to the correct space and then evaluate sample
        #     times by correct
        #
        #     if self._name == 'Gamma':
        #         do .....
        #
        #     if self._name == 'Exponential'
        #         do .....
        #
        #     """
        #     return  self.torch_dist.sample().type(torch.FloatTensor)
        # else:
        if sample_size:
            return self.torch_dist.sample(torch.Size([sample_size])).type(torch.FloatTensor)
        else:
            return self.torch_dist.sample(self._sample_shape).type(torch.FloatTensor)

    def log_pdf(self, x):
        # if self._transformed:
        #     """
        #     Transform x back to the right support and then evaluates log_pdf
        #     + jjacobian factor
        #     if self._name == 'Gamma':
        #         do .....
        #     return self.torch_dist.log_prob(x_constrained) + |J|
        #     if self._name == 'Exponential'
        #         do .....
        #
        #     """
        #     return self.torch_dist.log_prob(x)
        # else:
        x = VariableCast(x)
        return self.torch_dist.log_prob(x)
