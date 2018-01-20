from pyfo.pyfoppl.foppl import imports
import if_1d as test


print(test.code)
from pyfo.inference.dhmc import DHMCSampler as dhmc

dhmc_ = dhmc(test.model)
burn_in = 100
n_sample = 1000
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,plot=True, print_stats=True, save_samples=True)

samples =  stats['samples']
all_samples = stats['samples_wo_burin'] # type, panda dataframe