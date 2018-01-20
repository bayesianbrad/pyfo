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
    from pyfo.pyfoppl.foppl import imports
    import test_gauss_1d as test
    from pyfo.inference.dhmc import DHMCSampler as dhmc
    burn_in = 100
    n_samples = 1000
    stepsize_range = [0.05, 0.25]
    n_step_range = [10, 20]

    dhmc_ = dhmc(test.model)

    stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range, seed=123, print_stats=True)

    print('accept ratio: ', stats['accept_prob'])

def main():
    base_continous_test()

main()