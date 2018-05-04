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

n = 1
d = 1
x = sample(normal(torch.zeros(n,d), torch.ones(n,d)))
y = 1
observations = 7*torch.ones(n,d)
observe(normal(y, 2*torch.ones(n,d)), observations)
y
"""
model_if = """
import torch


x1  = sample(normal(torch.tensor([0,2]), torch.tensor([1,4])))

x2 = sample(normal(torch.tensor([0,5]), torch.tensor([2,4])))

observations = torch.ones(len(x2))


boolean = torch.gt(x1, 0)
truth_index = boolean.nonzero() # indices for which the statement is true
false_index = (boolean==0).nonzero() # indices for which the statements are false.

observe(normal(x2[boolean.nonzero()], 1*torch.tensor(len(x2[boolean.nonzero()]))),observations[boolean.nonzero()])
observe(normal(-1*torch.tensor(len(x2[(boolean==0).nonzero()])), torch.tensor(len(x2[(boolean==0).nonzero()]))), observations[(boolean==0).nonzero()])
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

zn = sample(categorical(pi))

for i in range(len(pi)):
    index = (zn == i).nonzero()
    observe(normal(mus[i]*torch.ones(len(index)), 2*torch.ones(len(index))), ys[index])
"""

neural_network = """
import torch

#
# latent_dim = 2
# hidden_dim = 10
# output_dim = 5
#
# def gaussian():
#     return sample(normal(0.0, 1.0))
#
# def make_latent_vector():
#     return [gaussian() for _ in range(latent_dim)]
#
# def make_hidden_vector():
#     return [gaussian() for _ in range(hidden_dim)]
#
# def make_output_vector():
#     return [gaussian() for _ in range(output_dim)]
#
# def relu(v):
#     return matrix.mul(matrix.ge(v, 0.0), v)
#
# def sigmoid(v):
#     return matrix.div(1.0, matrix.add(1.0, matrix.exp(matrix.sub(0.0, v))))
#
# def flip(i, p):
#     return sample(binomial(p[i]))
#
# z = make_latent_vector()
# W = [make_latent_vector() for _ in range(hidden_dim)]
# b = make_hidden_vector()
# h = relu(matrix.add(matrix.mmul(W, z), b))
#
# V = [make_hidden_vector() for _ in range(output_dim)]
# c = make_output_vector()
#
# result = []
# for i in range(output_dim):
#     result.append( flip(i, sigmoid(matrix.add(matrix.mmul(V, h), c))) )


latent_dim = 2
hidden_dim = 10
output_dim = 5

def gaussian(n_samples):
    return sample(normal(0.0*torch.ones(n_samples), 1.0*torch.ones(n_samples)))

def make_latent_vector():
    return gaussian(latent_dim)

def make_hidden_vector():
    return gaussian(hidden_dim)

def make_output_vector():
    return gaussian(output_dim)

def relu(v):
    relu = torch.nn.ReLU()
    return relu(v)

def sigmoid(v):
    return torch.sigmoid(v)

def flip(i, probs):
    return sample(binomial(total_count=i,probs=probs))

z = make_latent_vector()
W = torch.stack([make_latent_vector() for _ in range(hidden_dim)], dim=1) # Creates a tenssor of dims latent_dim x hidden_dim (2 x10)
b = make_hidden_vector() #(10)
h = relu(torch.mm(W.t(), z.unsqueeze(-1))+ b) # (10 x 2 * 2 x 1 + 10) --> 10 x 1

V = torch.stack([make_hidden_vector() for _ in range(output_dim)], dim=1) # 10 x 5
c = make_output_vector() # 5

result = []
# unclear from original model.
result.append( flip(1, probs=sigmoid(torch.mm(V.t(), h) + c)))"""




model_compiled = compile_model(model)
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

# x = model_compiled.gen_prior_samples()
print(model_compiled.gen_prior_samples())
# print(dir(model_compiled))
print(50 * '=')
model_compiled.is_torch_imported()

