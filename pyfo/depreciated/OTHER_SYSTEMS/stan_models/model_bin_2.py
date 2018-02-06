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

np.random.seed(1234)
import pystan


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
    real p;


}

parameters {

# real<lower=0, upper=1> x;
 real<lower=0, upper=1> z;
}

model {

    z ~ uniform(0,1);

    if (z<p) {
        y ~ normal(0,1);
      #  x ~ normal(0,0.00001);
    } else { 
        y ~ normal(1,1);
      #  x ~ normal(1,0.00001);
    }
}
'''

def initfun():
    return dict(y=0.25, p=0.5)
model = pystan.stan(model_code=model_bin_2, data=initfun(), iter=2000, chains=1)
print(model)
trace = model.extract()

plt.figure(figsize=(10, 4))
plt.hist(trace['z'][:], bins='auto', normed=1)
plt.savefig('model_2_z.png')
print('Completed plots')
# plt.figure(figsize=(10, 4))
# plt.hist(trace['x'][:], bins='auto', normed=1)
# plt.savefig('model_2_x.png')
print('Completed plots')
plt.figure(figsize=(10, 4))
plt.plot(trace['z'][:])
plt.savefig('model_2_z_trace.png')
print('Completed plots')
# plt.figure(figsize=(10, 4))
# plt.plot(trace['x'][:])
# plt.savefig('model_2_x_trace.png')
# print('Completed plots')