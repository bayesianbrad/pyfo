#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  22:48
Date created:  26/03/2018

License: MIT
'''

from pyfo.pyfoppl.pyppl import compile_model

import torch
# observed

n = torch.tensor(data=[54, 146, 169, 209, 220, 209, 250, 176, 172, 127, 123, 120, 142])
R = torch.tensor(data=[54, 143, 164, 202, 214, 207, 243, 175, 169, 126, 120, 120])
r = torch.tensor([24, 80, 70, 71, 109, 101, 108, 99, 70, 58, 44, 35])
m = torch.tensor([0,10, 37, 56, 53, 77, 112, 86, 110, 84, 77, 72, 95])
z = torch.tensor([0,14, 57, 71, 89, 121, 110, 132, 121, 107, 88, 60, 0])
u = torch.tensor(n-m)
T = len(n)

# priors
p = sample(uniform(torch.zeros(T),torch.ones(T)))
phi = sample(uniform( torch.zeros(T), torch.ones(T)))
U  = sample() # Probablematic as we can not represent 1/U_{i}

# first captures
lik1 = observe(binomial(U, p), u)

def initial_marked(phi, p, T):
    return 1 - phi(T-1)*p(T)
def conditional_marked(phi, p, chi, i):
    return 1 - phi(i)*(p(i+1) + (1 - p(i+1))(1 - chi(i+1)))

# re-captures
lik2 =