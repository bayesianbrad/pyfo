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
    chains_model3 = pickle.load(data)



means_gt_0_model3 = []
means_lte_0_model3 = []

for chain in chains_model3:
    means_gt_0_model3.append(np.mean([chain>0]))
    means_lte_0_model3.append(np.mean([chain<=0]))

print(' This is the expectaton > 0 for model 3 : {0} \n This i the expectation < 0 for model 3 : {1}'.format(means_gt_0_model3, means_lte_0_model3))

# model 2
with open('model_2_samples.pkl', 'rb') as data:
    chains_model2 = pickle.load(data)

means_gt_0_model2 = []
means_lte_0_model2 = []
q  =0.5
for chain in chains_model2:
    means_gt_0_model2.append(np.mean([chain>q]))
    means_lte_0_model2.append(np.mean([chain<=q]))

print(' This is the expectaton > 0 for model 3 : {0} \n This i the expectation < 0 for model 2 : {1}'.format(means_gt_0_model2, means_lte_0_model2))
