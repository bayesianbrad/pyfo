#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  20:52
Date created:  26/01/2018

License: MIT
'''
import torch
from pyfo.distributions.distribution_wrapper import TorchDistribution
import torch.distributions as dists
from pyfo.utils.core import VariableCast as vc
class LogGamma():
    """
    Implements a log gamma distribution

    """
    def __init__(self, concentration, rate, transformed=False):
        self.concentration = vc(concentration)
        self.rate = vc(rate)
        self._transformed = transformed


    def sample(self):
        u = dists.Uniform(vc(0),vc(1)).sample()
        return  torch.div(torch.log(u),self.concentration) + torch.lgamma(self.concentration) + torch.log(self.concentration)

    def log_pdf(self, x):
        """

        :param x: Assumes y: 'x' --> ln(x)
        :return:
        """
        x = vc(x)
        c1 = self.rate * x - torch.div(torch.exp(x),self.concentration)
        c2 = - self.rate*torch.log(self.concentration)
        c3 = - torch.lgamma(self.rate)
        return c1+c2+c3