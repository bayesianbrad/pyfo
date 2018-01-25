from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.ifmodels.nested_if as test


# print(test.model)
from pyfo.inference.dhmc import DHMCSampler as dhmc
test.model.display_graph()
dhmc_ = dhmc(test)
burn_in = 2999
n_sample = 2000
stepsize_range = [0.03,0.15]
n_step_range = [4, 15]

stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, plot=True, print_stats=True, save_samples=True, plot_ac=True)

# samples =  stats['samples']
# all_samples = stats['samples_wo_burin'] # type, panda dataframe