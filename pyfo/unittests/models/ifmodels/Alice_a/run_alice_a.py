from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.ifmodels.Alice_a.alice_a as test
from pyfo.inference.dhmc import DHMCSampler as dhmc
from pyfo.inference.bhmc import BHMCSampler as bhmc

# model
# print(test.model)
# test.model.display_graph()

# inference
dhmc_ = dhmc(test)
bhmc_ = bhmc(test)

burn_in = 2000
n_sample = 2000
stepsize_range = [0.03,0.15]
n_step_range = [4, 15]
n_chain = 10


# stats_dhmc = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,
#                      plot=True, print_stats=True, save_samples=True, plot_ac=True)
#
# stats_bhmc = bhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,
#                      plot=True, print_stats=True, save_samples=True, plot_ac=True)


stats_dhmc = dhmc_.sample_multiple_chains(n_chains=n_chain,n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,
                     plot=False, print_stats=False, save_samples=True, plot_ac=False)

stats_bhmc = bhmc_.sample_multiple_chains(n_chains=n_chain, n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range,
                     plot=False, print_stats=False, save_samples=True, plot_ac=False)