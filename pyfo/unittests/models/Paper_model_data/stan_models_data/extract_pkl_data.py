#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:32
Date created:  15/05/2018

License: MIT
'''
# open pickle
import pickle
import numpy as np

# model 3
with open('model_3_samples.pkl', 'rb') as data:
    chains_model4 = pickle.load(data)

print('{0} Printing for stan models 2 and 4 {0}\n'.format(5*'=='))


means_gt_0_model4 = []
means_lte_0_model4 = []
means_model4 = []
p = 0
for chain in chains_model4:
    means_gt_0_model4.append(np.mean(chain[chain>p]))
    means_model4.append(np.mean(chain))
    means_lte_0_model4.append(np.mean(chain[chain<=p]))
print(50*'-')
print(' This is the expectaton > {3} for model 4 : {0} \n This i the expectation < {3} for model 4 : {1} \n Overall means: {2}'.format(means_gt_0_model4, means_lte_0_model4, means_model4, p))
print(50*'-')
# model 2
with open('model_2_samples.pkl', 'rb') as data:
    chains_model2 = pickle.load(data)

means_gt_0_model2 = []
means_lte_0_model2 = []
means_model2 = []
q  =0.5
for chain in chains_model2:
    means_gt_0_model2.append(np.mean(chain[chain>q]))
    means_lte_0_model2.append(np.mean(chain[chain<=q]))
    means_model2.append(np.mean(chain))
print(50*'-')
print(' This is the expectaton > {3} for model 3 : {0} \n This i the expectation < {3} for model 2 : {1} \n The overall means are: {2}'.format(means_gt_0_model2, means_lte_0_model2, means_model2, q))
print(50*'-')