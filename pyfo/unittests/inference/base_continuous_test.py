#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:24
Date created:  09/01/2018

License: MIT
'''

import unittest
from pyfo.inference.dhmc import DHMCSampler as dhmc


def base_continous_test():
    from pyfo.inference.dhmc import DHMCSampler as dhmc
    burn_in = 100
    n_samples = 1000
    stepsize_range = [0.05, 0.25]
    n_step_range = [10, 20]

    dhmc_ = dhmc(model)

    stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range, seed=123)
    samples = stats[
        'samples']  # returns dataframe of all samples. To get all samples for a given parameter simply do: samples_param = samples[<param_name>]
    means = stats['means']  # returns dictionary key:value, where key - parameter , value = mean of parameter
    for key in means:
        print('{0} : {1}'.format(key, means[key]))

    print('accept ratio: ', stats['accept_prob'])

    print(samples['x20001'])