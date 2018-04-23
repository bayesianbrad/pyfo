#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  22:04
Date created:  23/04/2018

License: MIT
'''

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import scipy.stats as ss

import time
N = 10000
NBIN = 10

mean = 0
std = 1.414141
AVAILABLE_CPUS = mp.cpu_count()
chains = AVAILABLE_CPUS
# data = np.zeros(N+NBIN, 1, chains)
def sample_normal(N, NBIN):
    norm_dist = ss.multivariate_normal(mean=mean, cov=std)
    x = np.zeros(N+NBIN)
    for i in tqdm(range(N+NBIN)):
        x[i] = norm_dist.rvs()
    return x


if chains > 1:
    start1 = time.time()
    pool = mp.Pool(processes=AVAILABLE_CPUS)
    samples = [pool.apply_async(sample_normal, args=(N, NBIN)) for chain in range(chains-2)]
    samples = [chain.get() for chain in samples]
    end1 = time.time()

start2= time.time()
samples_1 = sample_normal(N,NBIN)
end2 = time.time()

print('Multiple chains time: {0} \n Single chain time: {1}'.format(end1-start1, end2- start2))


data =np.vstack(samples)
print(data.size)
print(data)


