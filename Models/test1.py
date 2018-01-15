#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  13:35
Date created:  29/12/2017

License: MIT
'''
# from pyfo.pyfoppl.foppl import imports
from onedgaussmodel import model
from pyfo.inference.dhmc2 import DHMCSampler as dhmc
# print(test.code)
# def test():
#     print(test.code)
#     print("=" * 50)
#     # burn_in = 100
#     model = test.model
#     cond_vars = model.gen_cond_vars()
#     cond_functions = model.cond_functions
#     state = model.gen_prior_samples()
#
#
#     for var in cond_vars:
#         print('this is a cond var')
#         print(state[var])
#     print(test.graph.get_parents_of_node(var))
#     print("=" * 50)
#     for key in model.cond_functions:
#         print(key,state[key])
#     print("=" * 50)
#     print('These are the if vars ')
#     print(test.model.gen_if_vars())
#     print("=" * 50)
#     print(test.graph)

burn_in = 10
n_samples = 10000
stepsize_range = [0.05,0.2]
n_step_range = [10, 20]

dhmc_ = dhmc(model)

stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range)
samples = stats['samples'] # returns dataframe of all samples. To get all samples for a given parameter simply do: samples_param = samples[<param_name>]
means = stats['means'] # returns dictionary key:value, where key - parameter , value = mean of parameter
for key in means:
    print('{0} : {1}'.format(key,means[key].data))

print('accept ratio: ', stats['accept_prob'])