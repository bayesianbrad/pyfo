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
model_if_nd = """
import torch


x1  = sample(normal(torch.tensor([[0,2]]), torch.tensor([[1,4]])))

x2 = sample(normal(torch.tensor([[0,5]]), torch.tensor([[2,4]])))

observations = torch.ones(x2.size())


boolean = torch.gt(x2, x1)
truth_index = boolean.nonzero() # indices for which the statement is true
false_index = (boolean==0).nonzero() # indices for which the statements are false.

observe(normal(x2[boolean.nonzero()], 1*torch.tensor(len(x2[boolean.nonzero()]))),observations[boolean.nonzero()])
observe(normal(-1*torch.tensor(len(x2[(boolean==0).nonzero()])), torch.tensor(len(x2[(boolean==0).nonzero()]))), observations[(boolean==0).nonzero()])
"""

model_if_1d ="""
x1 = sample(gamma(0.4, 0.3))
x2 = sample(normal(x1, 1))
y = 1
if x1> 0:
    observe(normal(x2, 1), y)
    observe(normal(-1,1), y)
x1,x2
"""

model_gmm2 = """
import torch

means  = 10
samples  = 20
y = [-2.0, -2.5, -1.7, -1.9, -2.2, 1.5, 2.2, 3.0, 1.2, 2.8,
      -1.7, -1.3,  3.2,  0.8, -0.9, 0.3, 1.4, 2.1, 0.8, 1.9] 
ys = torch.tensor(y)
pi = torch.tensor(0.5*torch.ones(samples,means))
mus = sample(normal(torch.zeros(means), 2*torch.ones(means)))

zn = sample(categorical(pi, size=10))

for i in range(len(pi)):
    index = (zn == i).nonzero()
    observe(normal(mus[i]*torch.ones(len(index)), 2*torch.ones(len(index))), ys[index])
"""

model_compiled = MCMC(model_code=model_gmm2, generate_graph=True, debug_on=True)
samples = model_compiled.run_inference(kernel=HMC,  nsamples=20, burnin=10, chains=4)