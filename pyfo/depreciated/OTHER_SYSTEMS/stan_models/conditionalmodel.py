#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  09:11
Date created:  26/09/2017

License: MIT
'''

import pickle
# import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1234)
import pystan

model_code =''' 
data {
    real y;
}

parameters {
      
      real x;

}

model {
     x ~ normal(1, 2.23606797749979);
    target += normal_lpdf(y | x,1.4142135623730951);
}
     
        
 '''
# set up the model
def initfun():
    return  dict(y=7)
model = pystan.stan(model_code=model_code, data=initfun(),iter=10000, chains=1)
print(model)

# trace = model.extract()
# print(trace)
# dat = trace['x'][:]
# plt.figure(figsize=(10,4))
# plt.hist(dat, bins ='auto', normed=1)
# plt.savefig('conditionalif.png')
print('Completed plots')
# # PyStan uses pickle to save objects for future use.
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model,f)
#
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)
#
# print(model)
