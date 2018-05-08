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

n = 10
d = 10
x1 = sample(normal(torch.zeros(n,d), torch.ones(n,d)))
x2 = sample(normal(x1, torch.ones(n,d)))
y = 7*torch.ones(n,d)
observe(normal(x1, 2*torch.ones(n,d)),y)
y
"""
model_compiled = MCMC(model_code=model, generate_graph=True, debug_on=False)


samples = model_compiled.run_inference(kernel=HMC,  nsamples=100, burnin=10, chains=4)
print(samples)