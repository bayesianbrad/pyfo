#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  20:48
Date created:  25/01/2018

License: MIT
'''
import unittest
from pyfo.inference.dhmc import DHMCSampler as dhmc


def base_bin_test():
    from pyfo.pyfoppl.foppl import imports
    import bin as test
    from pyfo.inference.dhmc import DHMCSampler as dhmc
    burn_in = 100
    n_samples = 1000
    stepsize_range = [0.05, 0.25]
    n_step_range = [10, 20]
    # test.model.display_graph()
    dhmc_ = dhmc(test)

    stats = dhmc_.sample(n_samples=n_samples, burn_in=burn_in, stepsize_range=stepsize_range, n_step_range=n_step_range, seed=123, print_stats=True)

def main():
    base_bin_test()

main()