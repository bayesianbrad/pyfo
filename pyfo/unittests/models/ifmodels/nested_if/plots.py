#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  20:26
Date created:  29/01/2018

License: MIT
'''
import pandas as pd

x = pd.read_csv('./data/chain_0_samples_after_burnin.csv', header=0)
print(x)