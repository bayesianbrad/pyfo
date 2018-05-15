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
import time
start = time.time()
"""A

FOPPL Code:
    (let [y 7
          p 0.5
          x (let [z (sample (uniform 0 1))]
              (if (< z p) 0 1))]
      (if (< x 1)
        (observe (normal x 1) y)
        (observe (normal x 1) y))
      x)
"""

model_2 = '''

data {
    real y;
    real q;


}

parameters {

 real<lower=0, upper=1> z;
}

model {

    z ~ uniform(0,1);

    if (z<q) {
        y ~ normal(0,1);
    } else { 
        y ~ normal(1,1);
    }
}
'''


model_3 ='''

data {
        real y;
        }
parameters {

real x;
}
model {
    x ~ normal(0,1);
    
    if (x > 0) {
        y ~ normal(x+1,1);
            }
    else {
    y ~ normal(x - 1,1);
    }
}
'''

def initfun():
    return dict(y=0.25, q=0.5)
# def initfun():
#     return dict(y=1)
model = pystan.StanModel(model_code=model_2)
fit = model.sampling(data=initfun(), iter=10000, warmup=1000, chains=1)
trace = fit.extract()['z']
end= time.time()
print(fit)

chains = [trace]
for i in range(4):
    fit = model.sampling(data=initfun())
    trace = fit.extract()['z']
    chains.append(trace)

with open('model_2_samples.pkl', 'wb') as f:
    pickle.dump(chains, f, protocol=pickle.HIGHEST_PROTOCOL)
fit.plot()

# print('The total time taken is : {0}'.format(end-start))
# # plt.figure(figsize=(10, 4))
# # plt.hist(trace['x'][:], bins='auto', normed=1)
# # plt.savefig('stan_1.png')
# print('Completed plots')
# # plt.figure(figsize=(10, 4))
# # plt.hist(trace['x'][:], bins='auto', normed=1)
# # plt.savefig('model_2_x.png')
# print('Completed plots')
# plt.figure(figsize=(10, 4))

# plt.figure(figsize=(10, 4))
# plt.plot(trace['x'][:])
# plt.savefig('model_2_x_trace.png')
# print('Completed plots')