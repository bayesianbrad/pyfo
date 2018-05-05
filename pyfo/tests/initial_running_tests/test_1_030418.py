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

n = 1
d = 1
x = sample(normal(torch.zeros(n,d), torch.ones(n,d)))
y = 1
observations = 7*torch.ones(n,d)
observe(normal(y, 2*torch.ones(n,d)), observations)
y
"""
model_compiled = MCMC(model_code=model, generate_graph=False, debug_on=True)


samples = model_compiled.run_inference(kernel=HMC,  nsamples=100, burnin=10, chains=1)
print(samples)