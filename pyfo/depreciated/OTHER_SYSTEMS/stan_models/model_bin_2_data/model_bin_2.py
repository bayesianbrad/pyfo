#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:23
Date created:  06/02/2018

License: MIT
'''
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
np.random.seed(1234)
import pystan
import pandas as pd
plt.style.use('ggplot')


"""A

FOPPL Code:
    (let [y 0.25
          p 0.5
          x (let [z (sample (uniform 0 1))]
              (if (< z p) 0 1))]
      (if (< x 1)
        (observe (normal x 1) y)
        (observe (normal x 1) y))
      x)
"""

model_bin_2 = '''

data {
    real y;
    real q;
}

parameters {
real<lower=0, upper=1> x;
}

model {
    x ~ uniform(0,1);

    if (q>x) {
        y ~ normal(1,1);
    } else { 
        y ~ normal(0,1);
    }
}
'''

def initfun():
    return dict(y=0.25, q=0.5)

append_data  = []
for i in range(1):
    model = pystan.stan(model_code=model_bin_2, data=initfun(), iter=2000, chains=1)
    trace = model.extract()
    df = pd.DataFrame(trace['x'][:], columns=['x'+str(i)])
    append_data.append(df)

append_data = pd.concat(append_data, axis=1)
append_data.to_csv('Trace_data_5.csv')
fig_width = 3.39  # width in inches
golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_height = fig_width*golden_mean # height in inches
plt.figure(figsize=(fig_width, fig_height))
# sns.distplot(trace['z'][:], bins='auto', norm_hist=True, kde=False)
# weights = np.ones_like(trace['z'][:]) / len(trace['z'][:])
# plt.hist(trace['z'][:], bins=50, weights=weights, color="blue", alpha=0.5, normed=False)
# plt.savefig('model_2_z.pdf')
# print('Completed plots')
# # plt.figure(figsize=(10, 4))
# # plt.hist(trace['x'][:], bins='auto', normed=1)
# # plt.savefig('model_2_x.png')
# print('Completed plots')
# plt.figure(figsize=(10, 4))
# sns.plot(trace['z'][:])
# plt.savefig('model_2_z_trace.png')
# print('Completed plots')
# plt.figure(figsize=(10, 4))
# plt.plot(trace['x'][:])
# plt.savefig('model_2_x_trace.png')
# print('Completed plots')