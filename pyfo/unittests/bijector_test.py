#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:00
Date created:  01/02/2018

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
