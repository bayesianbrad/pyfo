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

model="""
import torch

n = 100
d = 10
x1 = sample(normal(torch.zeros(n,d), torch.ones(n,d)))
y = 1*torch.zeros(n,d)
observe(normal(torch.zeros(n,d), torch.ones(n,d)), y)
y
"""
model_dependent="""

y
"""
model_compiled = MCMC(model_code=model, generate_graph=True, debug_on=False)


samples = model_compiled.run_inference(kernel=HMC,  nsamples=1000, burnin=10, chains=4)
print(samples)

import torch

n = 100
d = 10
x1 = sample(normal(torch.zeros(n,d), torch.ones(n,d)))
x2 = sample(normal(x1, torch.ones(n,d)))
y = 1*torch.zeros(n,d)
observe(normal(x2, torch.ones(n,d)), y)