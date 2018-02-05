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
#
model_code = '''
data {
    real y1;
    real y2;
}

parameters {

      real x;

}

model {
     x ~ normal(0, 1);
     if (x >= 0)
        if (x <1)
            target += normal_lpdf(y1 | x,1);
            if (x > 0.5)
                target += normal_lpdf(y1 | x, 1);
            else
                target += normal_lpdf(y1 | -x, 1);
        else
            target += normal_lpdf(y1 | x, 2);
     else
        target += normal_lpdf(y2 | x,3);
}


 '''

model_code = '''
data {
int y ;
}

parameters {

      real<lower=0, upper=1>  q;
      real<lower=0, upper=1>  z;
      # real<lower=0, upper=1> x;

}

# transformed parameters {
#  real tildeq;
# real  tildez;
# tildeq 
#     
# }
model {
     q ~ uniform(0, 1);
     z ~ uniform(0,1);
     
     
    
    {y ~ normal(z<q ? 0 : 1 ,1);}
        
    
    }


#  '''

# set up the model
def initfun():
    return dict(y=0)

#
model = pystan.stan(model_code=model_code, data=initfun(), iter=2000, chains=1)
print(model)
trace = model.extract()
# print(trace)
# # dat = trace['x'][:]
# # print(dat)
# # with open('trace.pkl', 'wb') as f:
# #     pickle.dump(dat,f)
# plt.figure(figsize=(10, 4))
# plt.hist(trace['y'][:], bins='auto', normed=1)
# plt.savefig('conif_y.png')
# print('Completed plots')
# # PyStan uses pickle to save objects for future use.
# with open('model.pkl', 'wb') as f:
#     pickle.dump(model,f)
#
# with open('trace.pkl', 'rb') as f:
#    trace = pickle.load(f)

# print(type(trace))
# print(trace)

# print(model)
# samples = model.extract()['x'][:]
# print(samples)
# lt_0 = (trace < 0)
# gte_0_lt1 = trace[(trace>= 0) & (trace<1)]
# gte_0_gte1 = (trace>=1)
#
# mean_lt  = np.mean(lt_0)
# mean_gte_0_lt1  = np.mean(gte_0_lt1)
# mean_gte_1  = np.mean(gte_0_gte1)
# print('Greater than 0 lt 1 expectation {0} and less than 0 expectation {1} and greater than 1 expectation  {2}'.format(mean_gte_0_lt1,mean_lt, mean_gte_1 ) )