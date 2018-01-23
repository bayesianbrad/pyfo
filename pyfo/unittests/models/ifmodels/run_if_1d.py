from pyfo.pyfoppl.foppl import imports
import if_1d as test


print(test.model)
from pyfo.inference.dhmc import DHMCSampler as dhmc

dhmc_ = dhmc(test)
burn_in = 10
n_sample = 100
stepsize_range = [0.03,0.15]
n_step_range = [1, 4]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,seed=123,plot=True, print_stats=True, save_samples=True)

samples =  stats['samples']
all_samples = stats['samples_wo_burin'] # type, panda dataframe
