#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pyfo.pyfoppl.foppl import imports
import pandas as pd
from pyfo.utils.eval_stats import *
from matplotlib import pyplot as plt

PATH  = sys.path[0] + '/data2018-01-30'
N = 10

n_chain = 2


### plot param
fontsize = 15
linewidth = 3

### load data
samples_DHMC = load_data(n_chain, PATH, True)
# reorder the keys
new_columns = {}
new_columns['get-ordered-mu.mu1'] = 0
new_columns['get-ordered-mu.mu2'] = 1
for i in range(N):
    new_columns['sample-components.zs_'+str(i)] = i+2
new_columns['zs_9'] = 11
# samples_DHMC_reorder: column 0 & 1 are mu1, mu2; columm 2-11 are z_0 to z_9
samples_DHMC_reorder = samples_DHMC
for i in range(n_chain):
    samples_DHMC_reorder[i].rename(columns=new_columns, inplace=True)
    samples_DHMC_reorder[i] = samples_DHMC_reorder[i][[j for j in range(N + 2)]]


### MCMC diagnotics:
# ess_mc = {}
# r_hat = {}
# all_vars = list(all_stats[0]['samples'])
# for key in all_vars:
#     n_sample, n_dim =  all_stats[0]['samples'].as_matrix(columns=[key]).shape
#     for i in range(n_chain):
#         samples = np.empty(shape=(n_chain, n_sample, n_dim))
#         samples[i] = all_stats[i]['samples'].as_matrix(columns=[key])
#
#     # ess = effective_sample_size(samples, mu_true, var_true)
#     # print('ess for {}: '.format(key), ess)
#     ess_mc[key] = effective_sample_size(samples)
#     r_hat[key] = gelman_rubin_diagnostic(samples)
#
# print('monte carlo ess: ', ess_mc)
# print('r value for: ', r_hat)


### other diagnotics by hand
# for i in range(n_chain):
#     # plt.hist(all_stats[i]['samples']['mus_0'], alpha = 0.2, bins='auto', normed=1)
#     plt.figure(1)
#     plt.hist(all_stats[i]['samples']['get-ordered-mu.mu1'], alpha=0.2, bins='auto', normed=1)
#     plt.figure(2)
#     plt.plot(all_stats[i]['samples']['get-ordered-mu.mu1'])
# plt.show()
#
# for i in range(n_chain):
#     # plt.hist(all_stats[i]['samples']['mus_0'], alpha = 0.2, bins='auto', normed=1)
#     plt.figure(1)
#     plt.hist(all_stats[i]['samples']['get-ordered-mu.mu2'], alpha=0.2, bins='auto', normed=1)
#     plt.figure(2)
#     plt.plot(all_stats[i]['samples']['get-ordered-mu.mu2'])
# plt.show()

### flip mu1 mu2
n_sample = 12000
n_burnin = 2000 # 10000
mu1 = np.empty(shape=(n_chain, n_sample))
mu2 = np.empty(shape=(n_chain, n_sample))

for i in range(n_chain):
    mu1[i] = samples_DHMC_reorder[i][0]
    mu2[i] = samples_DHMC_reorder[i][1]
    for j in np.where(mu1[i] > mu2[i]):
        temp = mu1[i][j]
        mu1[i][j] = mu2[i][j]
        mu2[i][j] = temp

for i in range(n_chain):
    # plt.hist(all_stats[i]['samples']['mus_0'], alpha = 0.2, bins='auto', normed=1)
    plt.figure(1)
    plt.hist(mu1[i][n_burnin:], alpha=0.2, bins='auto', normed=1)
    plt.figure(2)
    plt.plot(mu1[i][n_burnin:])

# plt.figure(1)
# plt.xlabel('$\mu_1$', fontsize=fontsize)
# plt.ylabel('$p(\mu_1|data)$', fontsize=fontsize)
# plt.yticks(size=fontsize)
# plt.xticks(size=fontsize)
# plt.savefig(PATH + '/mu1_hist.pdf')
# plt.figure(2)
# plt.xlabel('sample size', fontsize=fontsize)
# plt.ylabel('$\mu_1$', fontsize=fontsize)
# plt.yticks(size=fontsize)
# plt.xticks(size=fontsize)
# plt.savefig(PATH + '/mu1_trace.pdf')


for i in range(n_chain):
    # plt.hist(all_stats[i]['samples']['mus_0'], alpha = 0.2, bins='auto', normed=1)
    plt.figure(1)
    plt.hist(mu2[i][n_burnin:], alpha=0.2, bins='auto', normed=1)
    plt.figure(2)
    plt.plot(mu2[i][n_burnin:])
plt.figure(1)
plt.xlabel('$\mu_{1:2}$', fontsize=fontsize)
plt.ylabel('$p(\mu_{1:2}|data)$', fontsize=fontsize)
plt.yticks(size=fontsize)
plt.xticks(size=fontsize)
plt.savefig(PATH + '/mus_hist.pdf')
plt.figure(2)
plt.xlabel('sample size', fontsize=fontsize)
plt.ylabel('$\mu_{1:2}$', fontsize=fontsize)
plt.yticks(size=fontsize)
plt.xticks(size=fontsize)
plt.savefig(PATH + '/mus_trace.pdf')