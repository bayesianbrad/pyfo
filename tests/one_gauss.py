#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:27
Date created:  13/03/2018
License: MIT
'''
from pyfo.pyfoppl.pyppl import compile_model
model="""import torch
n = 50
d = 3
x = sample(normal(3*torch.ones(n,d), 5*torch.ones(n,d)))
# x = sample(normal(torch.zeros([784,10]), torch.ones([784,10])))
y = x + 1
observations = 7*torch.ones(n,d)
observe(normal(y, 2*torch.ones(n,d)), observations)
y
"""
model_if = """
import torch


x1  = sample(normal(torch.tensor([0,2]), torch.tensor([1,4])))

x2 = sample(normal(torch.tensor([0,5]), torch.tensor([2,4])))

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
observe(normal(x2[boolean.nonzero()], 1*torch.tensor(len(x2[boolean.nonzero()]))),observations[boolean.nonzero()])
observe(normal(-1*torch.tensor(len(x2[(boolean==0).nonzero()])), torch.tensor(len(x2[(boolean==0).nonzero()]))), observations[(boolean==0).nonzero()])
"""
model_compiled = compile_model(model)
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
true_names = {}
for vertex in vertices:
    true_names[vertex.name] = vertex.original_name
print('True names {}'.format(true_names))
