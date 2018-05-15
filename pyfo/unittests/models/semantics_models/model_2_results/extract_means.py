#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:10
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
p = 0.5
files =5
means_gt_0 = []
means_lte_0 = []
means = []
for i in range(files):
    file_name = 'chain' + str(i) + '_model2.csv'
    df = pd.read_csv(file_name)
    temp = df['x'][df['x'] > p]
    means_gt_0.append(temp.mean())
    temp = df['x'][df['x']<=p]
    means_lte_0.append(temp.mean())
    means.append(df.mean()[0])
print('{4} model_2 {4} \n the expectation means > {3} are : {0} \n The expectation means <= {3} are: {1} \n and overall means are: {2}'.format(means_gt_0, means_lte_0, means, p, 5*'='))

