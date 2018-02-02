import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('chain_0_all_samples.csv')
col = list(df)

def expected_value(key):
    score_lt1 =  df[key] > 0
    score_lt2 = df[key] <= 0
    # score_lt3 = df[key] == 2
    sum_of = score_lt1.sum()
    exp_0 = sum_of/len(df[col[0]])
    sum_of_2  =score_lt2.sum()
    exp_1 = sum_of_2 / len(df[col[0]])
    # sum_of_3  =score_lt3.sum()
    # exp_2 = sum_of_3 / len(df[col[0]])
    print('Expectation of {0} 0 : {1} '.format(key,exp_0))
    print('Expectation of {0} 1 : {1}'.format(key,exp_1))
    # print('Expectation of {0} 2 : {1}'.format(key,exp_2))
keys = ['x']
for key  in keys:
    expected_value(key)

df.hist(bins='auto', normed=1)
plt.show()