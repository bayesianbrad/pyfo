from pyfo.pyfoppl.foppl import imports
import mvn as test
# import gamma_test as model
# import matplotlib.pyplot as plt
# from pandas.compat import lmap
# import pandas as pd
# import numpy as np
# print(dir(test_if_src))
# print("=" * 50)
print(test.code)
print("=" * 50)
# print(test_if_src.graph)
# print("=" * 50)
# print(help(test_if_src.model))`

# from onedgaussmodel import model
from pyfo.inference.dhmc import DHMCSampler as dhmc



# # #
# burn_in = 0
# n_samples = 1000
# stepsize_range = [0.07,0.20]
# n_step_range = [10, 20]
# #
# bo = True
# dhmc_ = dhmc(test.model)
# # dhmc_ = dhmc(model)
# stats = dhmc_.sample(n_samples, burn_in, stepsize_range, n_step_range,seed=4, lag=50, print_stats=bo, plot=bo, plot_ac=bo)
# samples = stats['samples'] # returns dataframe of all samples. To get all samples for a given parameter simply do: samples_param = samples[<param_name>]
# sample_stats = stats['stats'] # returns dictionary key:value, where key - parameter , value = mean of parameter
# #import pandas as pd
# # for key in stats['param_names']:
# #     plot_pacf(samples[key], lags=50, label=key)
# # plt.legend()
# # plt.show()
# # import matplotlib.pyplot as plt
# print('accept ratio: ', stats['accept_prob'])
# #
# # means = samples.mean()
# # print(means)
# # names = stats['param_names']
# print(names)
