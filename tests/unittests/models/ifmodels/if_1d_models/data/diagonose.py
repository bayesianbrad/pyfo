#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  10:47
Date created:  05/02/2018

License: MIT
'''

import pandas as pd
import numpy as np

dataframe = pd.read_csv('dhmc_chain_0_all_samples.csv')

x=  dataframe['x'][1000:]

gte_0_lt1 = x[ (x>= 0) & (x<1)]
gte_lt1_gt_05 = x[(x>0.5) & (x<1)]
gte_lt1_lte_05 = x[ (x>0) & (x<=0.5)]
gte_1 = x>=1
lt_0 = x<0
gte_0 = x>=0
dict_exps = {'Mean of all x':x ,'Greater than equal to 0':gte_0, 'Greater than 0, less than 1': gte_0_lt1, 'Greater than equal to 1': gte_1, 'Less than 0': lt_0,  'Greater 0 less than equal 0.5': gte_lt1_lte_05, 'Greater 0 greater than 0.5 ': gte_lt1_gt_05}
def calc_means(x):
    return x.sum()/x.count()

for i in dict_exps:
    print('Expected value for x {0} is : {1} '.format(i, calc_means(dict_exps[i])))
#
# lt_0 = (trace < 0)
# gte_0_lt1 = trace[(trace>= 0) & (trace<1)]
# gte_0_gte1 = (trace>=1)

