#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:23
Date created:  05/02/2018

License: MIT
'''
import time
start = time.time()
from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.semantics_models.semantics as test
from pyfo.inference.dhmc import DHMCSampler as dhmc
dhmc_ = dhmc(test)

burn_in = 10 ** 3
n_sample = 10 ** 4

stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

stats,end_time = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, seed=122878, plot_graphmodel=True, print_stats=True,plot=True, save_samples=True)

print('Total run time : {}'.format(end_time-start))
# samples =  stats['samples']
# all_samples = stats['samples_wo_burin'] # type, panda dataframe
