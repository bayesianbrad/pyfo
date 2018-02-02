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
import torch.distributions.constraints as constraints
from torch.distributions.constraint_registry import biject_to
import torch.distributions as dists
import torch.distributions.transforms as transforms
from pyfo.utils.core import VariableCast as vc
from torch.distributions import TransformedDistribution as td
dist1 = dists.Exponential(vc(2))
exp_transfomed = td(dist1,[transforms.LogTransform(1)])
# transforms._InverseTransform(transform=transforms.ExpTransform)
sample =exp_transfomed.sample()
print(exp_transfomed.support)
print(sample)
logp = exp_transfomed.log_prob(sample)
print(logp)
unconstrained = sample
sample2 = biject_to(dist1.support)(unconstrained)
potential_energy = -dist1.log_prob(sample2).sum()
print(dist1.log_prob(sample2))
# unconstrained = Variable(torch.zeros(1), requires_grad=True)
# sample = biject_to(dist1.support)(constraints._GreaterThan(lower_bound=vc(0)))
# potential_energy = -dist.log_prob(sample).sum()
print(sample2)
print(potential_energy)
# constraint = Normal.params['scale']
# scale = transform_to(constraint)(torch.zeros(1))  # constrained
# u = transform_to(constraint).inv(scale)  # unconstrained

# print(potential_energy)