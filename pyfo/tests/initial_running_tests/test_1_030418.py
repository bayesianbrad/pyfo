#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  22:56
Date created:  03/05/2018

License: MIT
'''

import time

from pyfo.inference.mcmc import MCMC
from pyfo.inference.hmc import HMC

model_gamma="""
import torch
n = 1 
d = 1
x1 = sample(gamma(2*torch.ones(n,d), 3*torch.ones(n,d)))
# y = 1*torch.zeros(n,d)
# observe(normal(torch.zeros(n,d), torch.ones(n,d)), y)
x1
"""
model_normal="""
import torch 

n = 1 
d = 1
x1 = sample(normal(torch.zeros(n,d), 1*torch.ones(n,d)))
# y = 1*torch.zeros(n,d)
# observe(normal(torch.zeros(n,d), torch.ones(n,d)), y)
x1

"""
model_mvn="""
import torch 

n = 100
d = 1
x1 = sample(mvn(2*torch.ones(d,n), torch.eye(n)))
# y = 1*torch.zeros(n,d)
# observe(mvn(torch.zeros(n,d), torch.ones(n,d)), y)
x1
"""
model_mvnnorm_dependent="""
import torch 

n = 100
d = 1
x2 = sample(normal(torch.zeros(d,n), 1*torch.ones(d,n)))
x1 = sample(mvn(x2, torch.eye(n)))
y = torch.rand((n,d))
observe(mvn(x1, torch.eye(n)), y)
y
"""
model_dependent="""
import torch

n = 100
d = 10
x1 = sample(normal(torch.zeros(n,d), torch.ones(n,d)))
x2 = sample(normal(x1, torch.ones(n,d)))
y = torch.rand((n,d))
observe(normal(x2, torch.ones(n,d)), y)
y
"""
model_compiled = MCMC(model_code=model_gamma, generate_graph=True, debug_on=True)
samples = model_compiled.run_inference(kernel=HMC,  nsamples=20, burnin=10, chains=4)