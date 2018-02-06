#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:51
Date created:  26/09/2017

License: MIT
'''
import pickle
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1234)
import pystan

model_code = ''' 
data {
    real y1;
    real y2;
}

parameters {

      real x;

}

model {
     x ~ normal(0, 2);
     if (x >= 0)
        target += normal_lpdf(y1 | x+2,2);
     else
        target += normal_lpdf(y2 | x-2,2);
}


 '''


# set up the model
def initfun():
    return dict(y1=5, y2=-5)


model = pystan.stan(model_code=model_code, data=initfun(), iter=10000, chains=4)
print(model)

trace = model.extract()
print(trace)
dat = trace['x'][:]
plt.figure(figsize=(10, 4))
plt.hist(dat, bins='auto', normed=1)
plt.savefig('conif2.png')
print('Completed plots')
# # PyStan uses pickle to save objects for future use.
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model,f)
#
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
#
# print(model)
