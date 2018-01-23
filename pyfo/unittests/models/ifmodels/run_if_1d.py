from pyfo.pyfoppl.foppl import imports
import if_1d as test


print(test.code)
from pyfo.inference.dhmc import DHMCSampler as dhmc

dhmc_ = dhmc(test)
burn_in = 20000
n_sample = 50000
stepsize_range = [0.05,0.25]
n_step_range = [3, 10]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, plot=True, print_stats=True, save_samples=True, plot_ac=True)

samples =  stats['samples']
all_samples = stats['samples_wo_burin'] # type, panda dataframe
