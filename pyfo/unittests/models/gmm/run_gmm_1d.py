from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.gmm.gmm_1d_a as test
from pyfo.inference.dhmc import DHMCSampler as dhmc

# model
print(test.model)
test.model.display_graph()

# inference
# dhmc_ = dhmc(test)
#
# burn_in = 2000
# n_sample = 2000
# stepsize_range = [0.03,0.15]
# n_step_range = [10, 20]
# n_chain  = 5
#
# # stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,plot=False, print_stats=True, save_samples=True)
#
# all_stats = dhmc_.sample_multiple_chains(n_chains=n_chain, n_samples=n_sample,burn_in=burn_in,
#                                          stepsize_range=stepsize_range,n_step_range=n_step_range, save_samples=True)