#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:26
Date created:  13/03/2018
License: MIT
'''
from ..pyfoppl.pyppl import compile_model
model = """
import torch


x1  = sample(normal(torch.Tensor([0,2]), torch.Tensor([1,4])))

x2 = sample(normal(torch.Tensor([0,5]), torch.Tensor([2,4])))

observations = torch.ones(len(x2))
# this could get potentially tricky. As although we are performing an ''if'' statement over a vector, we have no
# explict 'if-then-else' statements.
# it may be wise to have, in addition to oberve and sample statements and if statement in which the user writes,
#  if x1 > 0:
#       do something
# else:
#       do something
# and unpack this within the model as follows

boolean = torch.gt(x1, 0)
truth_index = boolean.nonzero() # indices for which the statement is true
false_index = (boolean==0).nonzero() # indices for which the statements are false.

# These may be able to vectorized further
observe(normal(x2[boolean.nonzero()], 1*torch.Tensor(len(x2[boolean.nonzero()]))),observations[boolean.nonzero()])
observe(normal(-1*torch.Tensor(len(x2[(boolean==0).nonzero()])), torch.Tensor(len(x2[(boolean==0).nonzero()]))), observations[(boolean==0).nonzero()])
"""
# Of course if groups of indices have different bounds, this would get
# potentially very tricky. However,  we can ignore the 2nd case for now.

# orignal code

# x1 = sample(normal(0, 2))
# x2 = sample(normal(0, 4))
# if x1 > 0
#     observe(normal(x2, 1), 1)
# else:
#     observe(normal(-1, 1), 1)

model_compiled= compile_model(model, imports='import torch.distributions as dist')
print(model_compiled.code)
print(model_compiled.gen_cond_vars)
print('The if vars {}'.format(model_compiled.gen_if_vars()))
print('The cont vars {}'.format(model_compiled.gen_cont_vars()))
print('Get varaibles {}'.format(model_compiled.get_vars()))

vertices = model_compiled.get_vertices()
for vertex in vertices:
    if vertex.is_continuous:
        print(vertex.name)
print('Prior sample {}'.format(model_compiled.gen_prior_samples()))
print('Vertex names {}'.format(model_compiled.get_vertices_names()))