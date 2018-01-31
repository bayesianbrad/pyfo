#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  20:02
Date created:  30/01/2018

License: MIT
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('chain_0_samples_after_burnin.csv')
col = list(df)

score_lt1 =  df[col[0]] <= 0
score_lt2 = df[col[0]] > 0
sum_of = score_lt1.sum()
exp_s_g = sum_of/len(df[col[0]])
sum_of_2  =score_lt2.sum()
exp_s_ng = sum_of_2 / len(df[col[0]])
print(score_lt1)
print(score_lt2)
print(exp_s_g)
print(exp_s_ng)

df.hist(bins='auto', normed=1)
plt.show()

import scipy.stats as ss

normal = ss.norm()