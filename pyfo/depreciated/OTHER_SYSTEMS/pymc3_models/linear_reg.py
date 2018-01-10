#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:57
Date created:  02/11/2017

License: MIT
'''

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy import optimize

''' Model is Y ~ N(\mu, \sigma^{2})
             \mu = \alpha + \beta1 X1 + \beta2 X2
             
             \sigma - observation error
             
             Priors
             ------
             \alpha ~ N(0,100)  -weak prior
             \betai ~ N(0,100)  - weak prior
             \sigma ~ \abs{N(0,1)}
             
'''
np.random.seed(123)
def generate_data(datasize= 100, plot= False):
    ''' Returns randomly generated data'''
    # True parameter values
    alpha, sigma = 1, 1
    beta = [1,2.5]

    X1 = np.random.randn(datasize)
    X2 = np.random.randn(datasize)*0.2

    Y  = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(datasize)*sigma

    if plot:
        fig, axes = plt.subplots(1,2, sharex= True, figsize= (10,4))
        axes[0].scatter(X1,Y)
        axes[1].scatter(X2,Y)
        axes[0].set_ylabel('Y')
        axes[0].set_xlabel('X1')
        axes[1].set_xlabel('X2')
        plt.show()
    return Y, alpha, beta, X1, X2
def model_linear(Y, alpha, beta, X1, X2):
    '''
    Generates PyMC model.


    '''

    basic_model  = pm.Model() # creates a  new Model object which is a containor for the random variables.
    with basic_model:
        # Priors for unknown model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta  = pm.Normal('beta', mu=0, sd=10, shape = 2)
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value from Outcome
        mu = alpha + beta[0]*X1 + beta[1]*X2

        # Likilohood (sampling dist) of obsevations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

        map_estimate = pm.find_MAP(model= basic_model, method="POWELL")
        print('This is the MAP estimate ', map_estimate)
        trace = pm.sample()
        print('This is the trace ', trace)
        _ = pm.traceplot(trace)
        plt.show()
        print(pm.summary(trace))

def main():
    # Y, alpha, beta, X1, X2 = generate_data(plot=False)
    # modellr = model_linear(Y,alpha, beta, X1, X2)
    modelcoal = coal_mining_data()

main()