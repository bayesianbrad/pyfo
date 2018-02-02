from pyfo.pyfoppl.foppl import imports
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
# sns.set_context('paper')
# sns.set_style('white')
from pyfo.utils.eval_stats import *
import sys

PATH  = sys.path[0]
dir_DHMC = PATH + '/data2018-02-02T12:10:08.208509'
dir_BHMC = PATH + '/data2018-02-02T12:11:32.390353'

### load DHMC data
n_chain = 1
samples_DHMC_dict = load_data(n_chain, dir_DHMC,'dhmc', True)  #if true, load all data
samples_BHMC_dict = load_data(n_chain, dir_BHMC,'bhmc', True)
thr = 0  #the hand tune burnin

chain_num = 0

plt.figure(1)
samples_DHMC = samples_DHMC_dict[chain_num].as_matrix()
plt.hist(samples_DHMC, bins = 'auto', normed=1)
plt.show()

plt.figure(2)
samples_BHMC = samples_BHMC_dict[chain_num].as_matrix()
plt.hist(samples_BHMC, bins = 'auto', normed=1)
plt.show()

sample_size, dim = samples_DHMC.shape
samples_DHMC_ESS = np.empty(shape=(n_chain, sample_size, dim ))
samples_BHMC_ESS = np.empty(shape=(n_chain, sample_size, dim ))
for i in range(n_chain):
    samples_DHMC_ESS[i] = samples_DHMC_dict[i].as_matrix()
    samples_BHMC_ESS[i] = samples_BHMC_dict[i].as_matrix()
ESS_DHMC = effective_sample_size(samples_DHMC_ESS)
ESS_BHMC = effective_sample_size(samples_BHMC_ESS)
print(ESS_DHMC, ESS_BHMC)
