#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:25
Date created:  06/02/2018
License: MIT
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
data = []
means = {}
for i  in range(5):
    no = str(i)
    PATH = 'dhmc_chain_'+no+'.csv'
    df  =pd.read_csv(PATH)
    means[no] = df['z'].mean()
    data.append(df['z'])



data = pd.concat(data, axis=0).reset_index(drop=True)
print(data)
fig_width = 3.39  # width in inches
golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_height = fig_width*golden_mean # height in inches
plt.figure(figsize=(fig_width, fig_height))
sns.distplot(data[:], bins='auto', norm_hist=True, kde=False)
plt.savefig('histogram_dhmc.pdf')
data.to_csv('concatenated.csv')
print(means)
avg_mean = 0
for i in means:
    avg_mean += means[i]
avg_mean = avg_mean/len(means)

print(avg_mean)
np.save('means.npy', means)
print(np.std(list(means.values())))
