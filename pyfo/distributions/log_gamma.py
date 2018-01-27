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
class LogGamma(TorchDistribution):
    """
    Implements a log gamma distribution

    """
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate

    def sample(self):
        u = dists.Uniform(vc(0),vc(1))
        return  torch.div(torch.log(u)/self.concentration) + torch.lgamma(self.concentration) + torch.log(self.concentration)

    def log_prob(self, x):
        """

        :param x:
        :return:
        """
        x = vc(x)
        c1 = self.rate * x - torch.div(torch.exp(x),self.concentration)
        c2 = - self.rate*torch.log(self.concentration)
        c3 = - torch.lgamma(self.rate)
        return c1+c2+c3