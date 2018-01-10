#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:52
Date created:  02/11/2017

License: MIT
'''

import numpy as np
import pymc3 as pm
import theano
from theano import tensor as T
from pymc3.math import switch
import matplotlib.pyplot as plt


def coalmining():
    disaster_data = np.ma.masked_values([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                                3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                                2, 2, 3, 4, 2, 1, 3, -999, 2, 1, 1, 1, 1, 3, 0, 0,
                                1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                                0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                                3, 3, 1, -999, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                                0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1], value=-999)
    year = np.arange(1851, 1962)

    plt.plot(year, disaster_data, 'o', markersize=8);
    plt.ylabel("Disaster count")
    plt.xlabel("Year")

    with pm.Model() as disaster_model:

        switchpoint = pm.DiscreteUniform('switchpoint', lower=year.min(), upper=year.max(), testval=1900)

        # Priors for pre- and post-switch rates number of disasters
        early_rate = pm.Exponential('early_rate', 1)
        late_rate = pm.Exponential('late_rate', 1)

        # Allocate appropriate Poisson rates to years before and after current
        rate = pm.math.switch(switchpoint >= year, early_rate, late_rate)

        disasters = pm.Poisson('disasters', rate, observed=disaster_data)

    with disaster_model:
        trace = pm.sample(10000)

    pm.traceplot(trace)
    plt.show()

def model1():
    # priors
    # model1 = pm.Model()
    # with model1:
    #     x0 = pm.Normal('x0', mu=0, sd=1, shape=1)
    #     # if x0
    #         x1 = pm.Normal('x1',mu=-1, sd=1)
    #         y_obs = pm.Normal('y_obs', mu=x1, sd=1, observed=1)
    #     else:
    #         x2  =pm.Normal('x2',mu=1,sd=1)
    #         y_obs = pm.Normal('y_obs', mu=x2, sd=1, observed=1)
        trace = pm.sample()
        # print(trace)
    # _ = pm.traceplot(trace)
    # plt.show()
    # print(pm.summary(trace))



def main():
    N =  1
    t = np.arange(0,1)
    with pm.Model() as model:
        # prior
        x0 = pm.Normal('x0',mu=0,sd=1)
        #expected valuexx
        x1 = pm.Normal('x1',mu=-1, sd=1)
        x2 = pm.Normal('x2',mu=1, sd=1)

        # likelihood
        obs1 = pm.Normal('obs1', mu=x1, sd=1, observed=1)
        obs2 = pm.Normal('obs2', mu=x2, sd=1, observed=1)
        choice = pm.Deterministic('model1', T.switch(x0<0, obs1,obs2 ))
    with model:
        trace = pm.sample(2000)
    pm.traceplot(trace)
    plt.show()
main()