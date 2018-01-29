#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyfo.pyfoppl.foppl import imports
import pandas as pd
from pyfo.utils.eval_stats import *
from matplotlib import pyplot as plt

PATH  = sys.path[0]
n_chain = 5
var_key = ['mu1', 'mu2']#,  'sample-components.zs_0','sample-components.zs_1']

### load data
all_stats = load_data(n_chain,var_key,PATH)

### MCMC diagnotics:
ess_mc = {}
r_hat = {}
all_vars = list(all_stats[0]['samples'])
for key in all_vars:
    n_sample, n_dim =  all_stats[0]['samples'].as_matrix(columns=[key]).shape
    for i in range(n_chain):
        samples = np.empty(shape=(n_chain, n_sample, n_dim))
        samples[i] = all_stats[i]['samples'].as_matrix(columns=[key])

    # ess = effective_sample_size(samples, mu_true, var_true)
    # print('ess for {}: '.format(key), ess)
    ess_mc[key] = effective_sample_size(samples)
    r_hat[key] = gelman_rubin_diagnostic(samples)

print('monte carlo ess: ', ess_mc)
print('r value for: ', r_hat)


### other diagnotics by hand
for i in range(n_chain):
    # plt.hist(all_stats[i]['samples']['mus_0'], alpha = 0.2, bins='auto', normed=1)
    plt.figure(1)
    plt.hist(all_stats[i]['samples']['mu1'], alpha=0.2, bins='auto', normed=1)
    plt.figure(2)
    plt.plot(all_stats[i]['samples']['mu1'])
plt.show()

for i in range(n_chain):
    # plt.hist(all_stats[i]['samples']['mus_0'], alpha = 0.2, bins='auto', normed=1)
    plt.figure(1)
    plt.hist(all_stats[i]['samples']['mu2'], alpha=0.2, bins='auto', normed=1)
    plt.figure(2)
    plt.plot(all_stats[i]['samples']['mu2'])
plt.show()