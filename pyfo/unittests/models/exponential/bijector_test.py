#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  02:15
Date created:  30/01/2018

License: MIT
'''
import torch
from torch.autograd import Variable
import torch.distributions as dists
import torch.distributions.transforms as transforms
from pyfo.utils.core import VariableCast as vc
import torch.distributions.constraints as constraints
from torch.distributions.constraint_registry import biject_to
from torch.distributions import TransformedDistribution as td
exp_transfomed = td(dists.Exponential(vc(1)),[transforms.LogTransform(1)])
# transforms._InverseTransform(transform=transforms.ExpTransform)
dist =exp_transfomed.sample()
print(dist)
logp = exp_transfomed.log_prob(vc(-1))
print(logp)
# unconstrained = Variable(torch.zeros(1), requires_grad=True)
# sample = biject_to(dist.support)(unconstrained)
# potential_energy = -dist.log_prob(sample).sum()
# print(sample)
# print(potential_energy)