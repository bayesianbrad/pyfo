#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:34
Date created:  15/05/2018

License: MIT
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:07
Date created:  15/05/2018

License: MIT
'''

print('{0} Printing the dhmc results for model 4 {0} \n'.format(10*'-'))
p = 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
files =7
means_gt_0 = []
means_lte_0 = []
means =[]
for i in range(files):
    file_name = 'dhmc_chain_' + str(i+1) + '_samples_after_burnin.csv'
    df = pd.read_csv(file_name)
    temp = df['x'][df['x'] > p]
    means_gt_0.append(temp.mean())
    temp = df['x'][df['x']<=p]
    means_lte_0.append(temp.mean())
    means.append(df.mean()[0])
print(50*'-')
print('THe expectation means > {3} are : {0} \n The expectation means <= {3} are: {1}.\n The overall means are given as : {2}'.format(means_gt_0, means_lte_0, means, p))
print(50*'-')
