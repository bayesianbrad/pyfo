#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.gauss_1d as test
from pyfo.inference.dhmc import DHMCSampler as dhmc
from pyfo.utils.eval_stats import *
# print(test.code)
dhmc_ = dhmc(test)
burn_in = 10 ** 3
n_sample = 10 ** 3
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, print_stats=True,plot=True, save_samples=True)

samples =  stats['samples']
all_samples = stats['samples_wo_burin'] # type, panda dataframe


# samples = stats['samples']
# means = stats['means']
# print(means)
print(stats['accept_prob'])


x = samples.as_matrix(columns=['x'])
t,d = x.shape
x_ = np.empty(shape=(1,t,d))
x_[0] = x
mu_true = 5.28
std_true = 1.429
ess = effective_sample_size(x_, [mu_true], [std_true**2])
print(ess)
print('acceptace rate: ', acceptance_rate(x_) )