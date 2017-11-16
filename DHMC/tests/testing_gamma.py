#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:35
Date created:  15/11/2017

License: MIT
'''

from DHMC.distributions import *
import torch
from torch.autograd import Variable
torch.initial_seed()
print('This Gamma creates a fixed distribution class', Gamma)
print('This the gamma that takes parameters', gamma)

a = Variable(torch.Tensor([1]))
b = Variable(torch.Tensor([2]),requires_grad = True)
fixed = Gamma(a,b)
variable = Variable(gamma(a,b).data,requires_grad= True)
log_gamma_var = gamma.log_pdf(variable,a,b)
fixed_sample = Variable(fixed.sample().data, requires_grad= True)
log_gamma_fix =  fixed.log_pdf(x=fixed_sample)

grad1  = torch.autograd.grad(log_gamma_fix,[fixed_sample,b])
grad2  = torch.autograd.grad(log_gamma_var, [variable,b])
print('Fixed gradient {}'.format(grad1))
print('\nVariable gradient {}'.format(grad2))

print('\nsample fixed {}'.format(fixed_sample))
print('\bsample variable {}'.format(variable))



############