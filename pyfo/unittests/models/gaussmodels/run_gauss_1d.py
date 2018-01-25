#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.gauss_1d as test
from pyfo.inference.dhmc import DHMCSampler as dhmc
from pyfo.utils.eval_stats import *
# print(test.code)
dhmc_ = dhmc(test)
burn_in = 10 ** 1
n_sample = 10 ** 1
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]
n_chain = 2

all_stats = dhmc_.sample_multiple_chains(n_chain, n_sample, burn_in=burn_in, stepsize_range=stepsize_range, n_step_range=n_step_range)

for i in all_stats:
    print('Chain {}: acceptance rate {} \n'.format(i, all_stats[i]['accept_rate'] ))


### MCMC diagnotics:
# samples =  stats['samples']
# all_samples = stats['samples_wo_burin'] # type, panda dataframe
#
# means = stats['means']
# print(means)
# x = samples.as_matrix(columns=['x'])
# t,d = x.shape
# x_ = np.empty(shape=(n_batch,t,d))
# x_[0] = x
# x_[1] = x
# mu_true = [5.28]
# var_true = [1.429**2]
#
# ess = effective_sample_size(x_, mu_true, var_true)
# print(ess)
#
# r_hat = gelman_rubin_diagnostic(x_)
# r_truemean = gelman_rubin_diagnostic(x_, mu_true)
# print('r value: ', r_hat, r_truemean)
