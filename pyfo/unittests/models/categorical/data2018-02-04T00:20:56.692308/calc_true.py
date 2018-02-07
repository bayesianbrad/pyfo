#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  00:22
Date created:  04/02/2018

License: MIT
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  19:40
Date created:  03/02/2018

License: MIT
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('dhmc_chain_1.csv')
col = list(df)

z = [0,1,2]
for i in z:
    for j in col:
        print(j)
        score_lt1 = df[j] == i
        sum_of = score_lt1.sum()
        exp_1 = sum_of/len(df[col[0]])
        print(exp_1)

# df.hist(bins='auto', normed=1)
# plt.show()