#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:29
Date created:  13/03/2018
License: MIT
'''

from pyfo.pyfoppl.pyppl import compile_model

model="""import torch

# def observe_data(data, slope, bias):
#     xn = data[0]
#     yn = data[1]
#     zn = slope * xn + bias
#     observe(normal(zn, 1.0), yn)

# instead of a function we can write this in torch as follows:

slope = sample(normal(torch.tensor(0.0), torch.tensor(10.0)))
bias  = sample(normal(torch.tensor(0.0), torch.tensor(10.0)))
data  = torch.tensor([[1.0, 2.1], [2.0, 3.9], [3.0, 5.3]])
zn = slope*data[:,0] + bias # y  = mx + c
observe(normal(zn, torch.ones(len(zn))),data[:,1])

[slope, bias]
"""
model_code = compile_model(model)
print(model_code.code)