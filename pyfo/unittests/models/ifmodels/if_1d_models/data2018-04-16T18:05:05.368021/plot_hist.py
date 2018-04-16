#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  18:16
Date created:  16/04/2018

License: MIT
'''
import seaborn as sns
import pandas as pd
from  matplotlib import pyplot as plt
import numpy as np
x = pd.read_csv('sppl_samples_after_burnin.csv')
x = x.as_matrix()
# sns.distplot(x, kde=True)
# plt.figure(figsize=(10, 4))
# plt.hist(x['z'], bins='auto', normed=1)
num_bins = 100

fig, ax = plt.subplots()
# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, normed=True)

# add a 'best fit' line
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of SPPL inferred density')

# Tweak spacing to prevent clipping of ylabel
fig.savefig('sppl_7.pdf')
print('Completed plots')