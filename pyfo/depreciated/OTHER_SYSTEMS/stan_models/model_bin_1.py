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

"""
FOPPL model

(let [y 0.25
      p 0.5
      x (sample (bernoulli p))]
  (if (< x 1)
    (observe (normal x 1) y)
    (observe (normal x 1) y))
  x)
  
"""
model_bin_1 = '''

data {
real y;
real p;


}

parameters {
      real<lower=0, upper=1> x;
}


model {
    x ~ bernoulli(p);
    if (x < 1)
        target += normal_lpdf(y | x ,1 );
    else:
        target += normal_lpdf(y | x, 1);

}

'''

def initfun():
    return dict(y=0.25, p=0.5)

model = pystan.stan(model_code=model_bin_1, data=initfun(), iter=2000, chains=1)
print(model)
trace = model.extract()