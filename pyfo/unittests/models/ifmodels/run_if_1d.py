from pyfo.unittests.models.ifmodels.if_1d_model import model
from pyfo.inference.dhmc import DHMCSampler as dhmc
from pyfo.utils.eval_stats import *

dhmc_ = dhmc(model)
burn_in = 10 ** 1
n_sample = 10 ** 2
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, print_stats=True, save_samples=True)

samples =  stats['samples']
all_samples = stats['samples_wo_burin'] # type, panda dataframe

# print('mean_samples: ', extract_means(samples, 'x') , '\n')
# print('mean_samples w/o key: ', extract_means(samples) , '\n')
# print('mean_all_samples: ', extract_means(all_samples, 'x') , '\n')
# print(samples)
