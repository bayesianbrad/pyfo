from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.ifmodels.if_1d_models.if_1d as test
from pyfo.inference.bhmc import BHMCSampler as bhmc

# model
# print(test.model)
# test.model.display_graph()

# inference
bhmc_ = bhmc(test)
burn_in = 10
n_sample = 10
# stepsize_range = [0.03,0.15]
stepsize_range = [2,5] # force to cross the boundary
n_step_range = [4, 15]
n_chain = 5

stats = bhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,
                     plot=True, print_stats=True, save_samples=True)  # plot_ac=True

# all_stats = bhmc_.sample_multiple_chains(n_chains=n_chain, n_samples=n_sample,burn_in=burn_in,
#                                          stepsize_range=stepsize_range,n_step_range=n_step_range, save_samples=True)