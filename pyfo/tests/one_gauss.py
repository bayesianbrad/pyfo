#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:27
Date created:  13/03/2018
License: MIT
'''
import time
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

model_gmm2 = """
import torch

# def sample_components(pi):
#     return sample(categorical(pi))
#
# def observe_data(y, z, mus):
#     mu = mus[z]
#     observe(normal(mu, 2), y)
#
# ys = [-2.0, -2.5, -1.7, -1.9, -2.2, 1.5, 2.2, 3.0, 1.2, 2.8]
# pi = [0.5, 0.5]
# zs = map(sample_components, pi * 10)
# mus = [sample(normal(0, 2)), sample(normal(0, 2))]
# for y, z in interleave(ys, zs):
#     observe_data(y, z, mus)

means  = 10
samples  = 20
y = [-2.0, -2.5, -1.7, -1.9, -2.2, 1.5, 2.2, 3.0, 1.2, 2.8,
      -1.7, -1.3,  3.2,  0.8, -0.9, 0.3, 1.4, 2.1, 0.8, 1.9] 
ys = torch.tensor(y)
pi = torch.tensor(0.5*torch.ones(samples,means))
mus = sample(normal(torch.zeros(means), 2*torch.ones(means)))

zn = sample(categorical(pi))

for i in range(len(pi)):
    index = (zn == i).nonzero()
    observe(normal(mus[i]*torch.ones(len(index)), 2*torch.ones(len(index))), ys[index])
"""


model_compiled = compile_model(model_gmm2)
st = time.time()
print(model_compiled.code)
end = time.time()
print(' The time taken is {}'.format(end - st))
time.sleep(1)
print(model_compiled.gen_cond_vars)
print('The if vars {}'.format(model_compiled.gen_if_vars()))
print('The cont vars {}'.format(model_compiled.gen_cont_vars()))
print('The disc vars {}'.format(model_compiled.gen_disc_vars()))
print('Get varaibles {}'.format(model_compiled.get_vars()))

vertices = model_compiled.get_vertices()
for vertex in vertices:
    if vertex.is_continuous:
        print(vertex.name)
    dir(model_compiled)

print('Vertex names {}'.format(model_compiled.get_vertices_names()))
true_names = {}
for vertex in vertices:
    true_names[vertex.name] = vertex.original_name
print('True names {}'.format(true_names))
distribtuion_params = {}
# for vertex in vertices:
#     distribtuion_params[vertex.name] = vertex.distribution_arguments

for vertex in vertices:
    if vertex.is_sampled:
        distribtuion_params[vertex.name] = {vertex.distribution_name : vertex.distribution_arguments}

print('Distribution arguments : {}'.format(distribtuion_params))
print(50*'=')
print(distribtuion_params.values())
print(distribtuion_params.keys())
print(50*'=')
for param in model_compiled.get_vars():
    print(param)
    print(distribtuion_params[param].values())
    print(distribtuion_params[param].keys())
    print(50*'=')


print(model_compiled.gen_prior_samples())
# print(dir(model_compiled))
print(50 * '=')

