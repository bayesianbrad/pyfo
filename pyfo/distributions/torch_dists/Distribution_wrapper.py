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

    def __init__(self, torch_dist):
        super(TorchDistribution, self).__init__()
        self.torch_dist = torch_dist
        self._sample_shape = torch.Size()


    def sample(self):
        if self.reparameterized:
            return self.torch_dist.rsample(self._sample_shape)

    def log_pdf(self, x):
        x = VariableCast(x)
        return self.torch_dist.log_prob(x)
