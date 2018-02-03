# Last Modify: 01-02-2018
import math
import time
from itertools import permutations

import numpy as np
import pandas as pd
import torch
import copy
from torch.autograd import Variable
# the state interacts with the interface, where ever that is placed....
from pyfo.utils import state
from pyfo.utils.core import VariableCast
from pyfo.utils.eval_stats import extract_stats
from pyfo.utils.eval_stats import save_data
from pyfo.utils.plotting import Plotting as plot

class BHMCSampler(object):
    """
    In general model will be the output of the foppl compiler, it is not entirely obvious yet where this
    will be stored. But for now, we will inherit the model from pyro.models.<model_name>
    """

    def __init__(self, object, scale=None):
        # Note:
        ## Need to deal with a M matrix. Using the identity matrix for now.

        self.model_graph =object.model # i graphical model object
        self._state = state.State(self.model_graph)

        ## Debugging:::
        #####
        # print('Debug flug in BHMC sampler is one \n')
        # self._state.debug()

        # Parameter keys
        self._disc_keys = self._state._return_disc_list()
        self._cont_keys = self._state._return_cont_list() # cond_var
        self._cond_keys = self._state._return_cond_list()
        self._cond_map = self._state.get_conds_map()  # dict {if_var: corresponding cond_var}

        self._disc_temp_keys = self._disc_temp_keys.copy()
        self._cont_temp_keys = self._cont_temp_keys.copy()
        # By construction all keys that are predicates and are continous are put into if_keys, if predicate is dicont-
        # inuous then it is automatically goes into the discrete keys
        self._if_keys = self._state._return_if_list()   # if_var
        # True latent variable names
        self._names = self._state.get_original_names()
        # Generate sample sizes
        self._sample_sizes = self._state.get_sample_sizes()

        self._branch = False # If we cross a discontinuity whilst in the predicate this flag swtiches to true.
        self.grad_logp = self._state._grad_logp
        self.init_state = self._state.intiate_state() # this is just x0
        self.M = 1 #Assumes the identity matrix and assumes everything is 1d for now.

        self.log_posterior = self._state._log_pdf  # returns a function that is the current state as argument

    def random_momentum(self, branch=False, sign_p=None):
        """
        Constructs a momentum dictionary where for the discrete keys we have laplacian momen tum
        and for continous keys we have gaussian
        :return:
        #sign_p will only be none zero if we have branched
        """
        p = {}
        if self._disc_keys is not None:
            for key in self._disc_keys:
                p[key] = self.M * VariableCast(np.random.laplace(size=self._sample_sizes[key])) #in the future add support for multiple dims
        if self._cont_keys is not None:
            for key in self._cont_keys:
                p[key] = VariableCast(self.M * np.random.randn(self._sample_sizes[key]))
        if branch:
            for key in self._if_keys:
                if key in sign_p:
                    p[key] = self.M * sign_p[key] #will be + or - 1
                    # print('Debug statement in random_momentum() \n '
                    #       'for key : {0} the momentum is sign_p {1}'.format(key, sign_p[key]))
                else:
                    p[key] = self.M * VariableCast(np.random.laplace(size=self._sample_sizes[key]))
        if self._if_keys is not None and branch is False :
            for key in self._if_keys:
                p[key] = VariableCast(self.M * np.random.randn(self._sample_sizes[key]))
        return p

    def coordInt(self,x,p,stepsize,key, unembed=False):  # name conflict!! unembed as arg to partial_unembed or the actual unembed in method call
        """
        Performs the coordinate wise update. The permutation is done before the
        variables are executed within the function.

        :param x: Dict of state of positions
        :param p: Dict of state of momentum
        :param stepsize: Float
        :param key: unique parameter String
        :param unembed: type: bool
        :return: Updated x and p indicies
        """

        x_star = copy.copy(x)
        x_star[key] = x_star[key] + stepsize*self.M*torch.sign(p[key]) # Need to change M here again
        logp_diff = self.log_posterior(x_star, set_leafs=False, partial_unembed=unembed, key=key) \
                    - self.log_posterior(x, set_leafs=False, partial_unembed=unembed, key=key)
        # If the discrete parameter is outside of the support, returns -inf and breaks loop and integrator.
        if math.isinf(logp_diff.data[0]):
            return x[key], p[key], logp_diff.data[0]

        cond = torch.gt(self.M*torch.abs(p[key]),-logp_diff)
        if cond.data[0]:
            p[key] = p[key] + torch.sign(p[key])*self.M*logp_diff
            return x_star[key], p[key], 0
        else:
            p[key] = -p[key]
            return x[key],p[key], 0

    def append_keys(self, branch=False):
        """
        Creates a temp_dict of previous continuous variables that have to be added to the discrete
        variables
        :param branch: Type: bool, YZ: always pass in self._branch


        ### NOTES: weird for now. Add all if keys to disc_temp_keys
        for future: should only add those related to the boundary being crossed
        """
        ### old
        # if  branch:
        #     if self._disc_keys is not None:
        #         self._disc_temp_keys = self._disc_key + self._if_keys
        #     else:
        #         self._disc_temp_keys = self._if_keys
        # elif self._if_keys is not None:
        #         self._cont_temp_keys = self._if_keys
        #         if self._cont_keys is not None:
        #             self._cont_temp_keys = self._cont_temp_keys + self._cont_keys
        # else:
        #     self._cont_temp_keys = self._cont_keys

        ### new
        if not branch: # init;
            if self._cont_keys is not None:
                if self._if_keys is not None:
                    self._cont_temp_keys = self._cont_keys.copy() + self._if_keys.copy()
                else:
                    self._cont_temp_keys = self._cont_keys.copy()
            else:  # all cont are in ifs
                self._cont_temp_keys = self._if_keys.copy()
        else: # branch =  True, cross boundary
            if self._cont_temp_keys is not None:
                [self._cont_temp_keys.remove(key) for key in self._if_keys] # remove all for now, should only remove specific ones.
            if self._disc_temp_keys is not None:
                self._disc_temp_keys = self._disc_temp_keys + self._if_keys
                self._disc_temp_keys = list(set(self._disc_temp_keys))
            else: # no disc vars previously
                self._disc_temp_keys = self._if_keys
                self._disc_temp_keys = list(set(self._disc_temp_keys))

    def branch_check(self, x, x0, key ):
        """

        :param x:
        :param x0:
        :param key:
        :return:
        """
        if x[self._cond_map[key]] != x0[self._cond_map[key]]:
            # print('Debug statement in bhmc.branching_integrator(), '
            #       'the discontinuity has been crossed')
            # discontinuity has been crossed
            self._branch = True
            _branch = True
        else:
            _branch = False

        return  _branch

    def branching_integrator(self, xt, x0, p0, stepsize, t):
        """
        Performs the full DHMC update step. It updates the continous parameters using
        the standard integrator and the discrete parameters via the coordinate wie integrator.

        :param xt: Type dictionary, the pre ### YZ: not sure what it is for ? the same as x0?
        :param x0: Type dictionary, the previous state
        :param p: Type dictionary
        :param stepsize: Type: Variable
        :param t Type: int trajectory number
        :return: x, p the proposed values as dict.
        """

        # number of function evaluations and fupdates for discrete parameters
        n_feval = 0
        n_fupdate = 0
        if t > 0 and self._branch:   # YZ: may be useful if self._branch is True
            x0 = copy.copy(xt)
        # performs shallow copy
        x = copy.copy(xt)
        p = copy.copy(p0)
        # perform first step of leapfrog integrators
        if self._cont_temp_keys:  # ALL KEYS: include cont or if
            logp = self.log_posterior(x, set_leafs=True)
            for key in self._cont_temp_keys:
                p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])

        if self._disc_temp_keys is None:
            if self._if_keys is None:
                # All keys are cont_key, perform normal HMC
                x, p = self._hmc_cont_update(x,p,stepsize)
                return x, p, 0, n_feval, n_fupdate, 0
            else:
                # All keys: if, maybe cont
                x, p ,sign_p = self._branch_cont(x,p,x0,p0,stepsize)
                for key in self._cont_temp_keys:
                    x[key] = x[key] + stepsize * self.M * p[key]  # full step for postions
                    _ = self.log_posterior(x, set_leafs=True)  # YZ: 1st bug x[self._cond_map[key]] will not update automatically
                    if self.branch_check(x,x0,key):
                        sign_p = {key: torch.sign(p[key])}  # Will rewrite the dictionary each time disc is crossed
                        return x0, p0, sign_p, n_feval, n_fupdate, 0
                self._branch = False
                logp = self.log_posterior(x, set_leafs=True)
                for key in self._cont_temp_keys:
                    p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])
                return x, p, 0, n_feval, n_fupdate, 0

        else:  # have disc keys, maybe if keys
            # print('Evaluated with the coordinate wise integrator ')
            permuted_keys_list = []
            # if self._disc_temp_keys is not None:
            permuted_keys_list = permuted_keys_list + self._disc_temp_keys
            permuted_keys = permutations(permuted_keys_list, 1)
            # permutates all keys into one permutated config. It deletes in memory as each key is called
            # returns a tuple ('key', []), hence to call 'key' requires [0] index.

            if self._cont_keys is not None: #DO hmc step
                for key in self._cont_keys:
                    x[key] = x[key] + 0.5 * stepsize * self.M * p[key]

            for key in permuted_keys:
                if self._disc_keys is not None:
                    if key[0] in self._disc_keys:  #YZ
                        x[key[0]], p[key[0]], _ = self.coordInt(x, p, stepsize, key[0], unembed=True)
                else:
                    x[key[0]], p[key[0]], _ = self.coordInt(x, p, stepsize, key[0], unembed=False)
                if math.isinf(_):
                    return x0, p, 0, n_feval, n_fupdate, _
            n_fupdate += 1

            if self._cont_temp_keys is None:
                # have only disc and if keys
                for key in self._cont_keys:
                    x[key] = x[key] + stepsize * self.M * p[key]  # final full step for postions
                logp = self.log_posterior(x, set_leafs=True)
                for key in self._cont_keys:
                    p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])
                return x, p, 0, n_feval, n_fupdate, 0
            else:  # have if keys
                x, p, sign_p = self._branch_2(x,p,x0,p0,stepsize)
                return x, p, sign_p,  n_feval, n_fupdate, 0

    def _hmc_cont_update(self, x,p, stepsize):
        """

        :param x:
        :param p:
        :param stepsize:
        :return:
        """
        for key in self._cont_keys:
            x[key] = x[key] + stepsize * self.M * p[key]  # full step for postions
        logp = self.log_posterior(x, set_leafs=True)
        for key in self._cont_keys:
            p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])
        return x, p

    def _branch_cont(self, x, p, x0, p0, stepsize):
        """

        :param x:
        :param p:
        :param xo:
        :param po:
        :param stepsize:
        :return:
        """
        for key in self._cont_temp_keys:
            x[key] = x[key] + stepsize * self.M * p[key]  # full step for postions
            _ = self.log_posterior(x,
                                   set_leafs=True)  # YZ: 1st bug x[self._cond_map[key]] will not update automatically
            if self.branch_check(x, x0, key):
                sign_p = {key: torch.sign(p[key])}  # Will rewrite the dictionary each time disc is crossed
                x= x0
                p= p0
            else:
                self._branch = False
                logp = self.log_posterior(x, set_leafs=True)
                for key in self._cont_temp_keys:
                    p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])

        return x, p, sign_p

    def _branch_1(self,x, p, x0, p0, stepsize):
        """

        :param x:
        :param p:
        :param x0:
        :param p0:
        :param stepsize:
        :return:
        """
        if self._if_keys is None:
            # All keys are cont_key, perform normal HMC
            sign_p = 0
            for key in self._cont_keys:
                x[key] = x[key] + stepsize * self.M * p[key]  # full step for postions
            logp = self.log_posterior(x, set_leafs=True)
            for key in self._cont_keys:
                p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])

        else:
            # All keys: if, maybe cont
            for key in self._cont_temp_keys:
                x[key] = x[key] + stepsize * self.M * p[key]  # full step for postions
                _ = self.log_posterior(x,
                                       set_leafs=True)  # YZ: 1st bug x[self._cond_map[key]] will not update automatically
                if x[self._cond_map[key]] != x0[self._cond_map[key]]:
                    # print('Debug statement in bhmc.branching_integrator(), '
                    #       'the discontinuity has been crossed')
                    # discontinuity has been crossed
                    self._branch = True
                    sign_p = {key: torch.sign(p[key])}  # Will rewrite the dictionary each time disc is crossed
                    return x0, p0, sign_p, n_feval, n_fupdate, 0
            self._branch = False
            logp = self.log_posterior(x, set_leafs=True)
            for key in self._cont_temp_keys:
                p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])
            return x, p, 0, n_feval, n_fupdate, 0
        return x, p, sign_p
    def _branch_2(self, x, p, x0, p0,stepsize):
        """

        :param x:
        :param p:
        :param x0:
        :param p0:
        :return:
        """
        for key in self._if_keys:
            x[key] = x[key] + stepsize * self.M * p[key]  # final  full step for postions
            _ = self.log_posterior(x, set_leafs=True)
        x[key] = x[key] + stepsize * self.M * p[key]  # final  full step for postions
        _ = self.log_posterior(x, set_leafs=True)
        if x[self._cond_map[key]] != x0[self._cond_map[key]]:
            # discontinuity has been crossed
            # print('Debug statement in bhmc.branching_integrator()\n'
            #       'the discontinuity has been crossed')
            sign_p = {key: torch.sign(p[key])}
            self._branch = True
            x = x0
            p = p0
            sign_p= sign_p

        else:
            self._branch = False
            logp = self.log_posterior(x, set_leafs=True)
            sign_p = 0
            if self._cont_keys is not None:
                for key in self._cont_keys:
                    x[key] = x[key] + 0.5 * stepsize * self.M * p[key]

            if self._cont_temp_keys is not None:
                for key in self._cont_temp_keys:
                    p[key] = p[key] + 0.5 * stepsize * self.grad_logp(logp, x[key])
        return x, p, sign_p

    def _energy(self, x, p, branch=False):

        """
        Calculates the hamiltonian  #YZ: shouldn't we just use cont and disc key, since if keys have been appended!
        :param x:
        :param p:
        :param branch: false when break out and calculate the final energy using Guass; True when start the next iteration, and calculate the initial energy using Laplace
        :return: Tensor
        """
        if self._disc_keys is not None:
            kinetic_disc = torch.sum(torch.stack([self.M * torch.abs(p[name]) for name in self._disc_keys]))
        else:
            kinetic_disc = VariableCast(0)
        if self._cont_keys is not None:
            kinetic_cont = 0.5 * torch.sum(torch.stack([self.M * torch.dot(p[name], p[name]) for name in self._cont_keys]))
        else:
            kinetic_cont = VariableCast(0)
        if branch:
            kinetic_if = torch.sum(torch.stack([self.M * torch.abs(p[name]) for name in self._if_keys]))
        elif self._if_keys is not None:
            kinetic_if = torch.sum(torch.stack([self.M * torch.dot(p[name], p[name]) for name in self._if_keys]))
        else:
            kinetic_if = VariableCast(0)
        kinetic_energy = kinetic_cont + kinetic_disc + kinetic_if
        potential_energy = -self.log_posterior(x)

        return self._state._return_tensor(kinetic_energy) + self._state._return_tensor(potential_energy)

    def hmc(self, stepsize, n_step, x0, sign_p):
        """

        :param stepsize_range: List
        :param n_step_range: List
        :param x0:
        :param sign_p : dict If self_branch = True sign_p = {'cond_name': sign(p^{s-1}[key])
        :return:
        """

        p = self.random_momentum(branch=self._branch, sign_p=sign_p)
        intial_energy = self._energy(x0,p, branch=self._branch)
        n_feval = 0
        n_fupdate = 0
        x = copy.copy(x0)
        # if branch was trigger, i.e the condition of the predicate has been changed, we will break out of the
        # leapfrog and create a new dict of temporary dict of discrete vars. If there are no if stateemnts and
        # there are cont keys, returns self.cont)temp_keys == self.
        _branch = self._branch
        self.append_keys(branch=self._branch) # initial append, separate into cont_temp and disc_temp
        for i in range(n_step):
            x,p, sign_p, n_feval_local, n_fupdate_local, _ = self.branching_integrator(x,x0,p,stepsize,t=i)
            if math.isinf(_):
                break
            if self._branch:
                n_feval += n_feval_local
                n_fupdate += n_fupdate_local
                _branch = False   # YZ: used in calculating energy, need to use the same Kinetic form
                self.append_keys(branch=self._branch)
                break
            else:
                n_feval += n_feval_local
                n_fupdate += n_fupdate_local
        final_energy = self._energy(x,p, branch=_branch)
        acceptprob  = torch.min(torch.ones(1),torch.exp(final_energy - intial_energy)) # Tensor
        accept = 1
        if acceptprob[0] < np.random.uniform(0,1):
            x = x0
            accept = 0
        return x, sign_p, accept, n_feval, n_fupdate

    def sample(self, chain_num=0, n_samples=1000, burn_in=1000, stepsize_range=[0.05, 0.20], n_step_range=[5, 20],
               seed=None, n_update=10, lag=20,
               print_stats=False, plot=False, plot_graphmodel=False, save_samples=False, plot_burnin=False,
               plot_ac=False):        # Note currently not doing anything with burn in
        '''
        :param chain_num: the number of the chain the sampler is in
        :param n_samples:  number of samples to draw
        :param burn_in: number of burn in samples, discard by default
        :param stepsize_range:
        :param n_step_range: trajectory length
        :param seed: seed for generating random numbers, None by default
        :param n_update: number of printing statement
        :param lag: plot parameter
        :param print_stats: print details of sampler
        :param plot: whether generate posterior plots, False by default
        :param plot_graphmodel: whether generate GM plot, False by default
        :param save_samples: whether save posterior samples, False by default
        :param plot_burnin: plot parameter
        :param plot_ac: plot parameter
        :return: stats: dict of inference infomation. stats{'samples': df of posterior samples}
        '''

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            print('seed cecilia')
        x = self.init_state
        n_per_update = math.ceil((n_samples + burn_in)/n_update)
        n_feval = 0
        n_fupdate = 0
        x_dicts = []
        accept =[]

        tic = time.process_time()
        print(50*'=')
        print('The sampler is now performing inference....')
        print(50*'=')
        i = 0
        times_branched = 0
        sign_p = 0 #either a dict of keys that crossed branch or 0
        while i < n_samples+burn_in:
            stepsize = VariableCast(np.random.uniform(stepsize_range[0], stepsize_range[1])) #  may need to transforms to variables.
            n_step = np.ceil(np.random.uniform(n_step_range[0], n_step_range[1])).astype(int)
            x, sign_p, accept_prob, n_feval_local, n_fupdate_local = self.hmc(stepsize,n_step,x, sign_p)
            if self._branch:
                times_branched += 1
                i = i
            else:
                n_feval += n_feval_local
                n_fupdate += n_fupdate_local
                accept.append(accept_prob)
                x_numpy = self._state._unembed(copy.copy(x)) # There should be a quicker way to do this at the very end,
                #  using the whole dataframe series. It will require processing a whole byte vector
                x_dicts.append(self._state.convert_dict_vars_to_numpy(x_numpy))
                i += 1
            if (i + 1) % n_per_update == 0:
                print('{:d} iterations have been completed.'.format(i + 1))
        toc = time.process_time()
        time_elapsed = toc - tic
        print(50*'=')
        print('Number of branching evaluations: {0} \n'
              'as a proportion of total evaluations {1} '.format(times_branched, times_branched/(n_samples+burn_in)))
        print(50 * '=')
        n_feval_per_itr = n_feval / (n_samples + burn_in)
        n_fupdate_per_itr = n_fupdate / (n_samples + burn_in)
        if self._disc_keys is not None and self._if_keys is not None:
            print('Each iteration of DHMC on average required '
                + '{:.2f} conditional density evaluations per discontinuous parameter '.format(n_fupdate_per_itr / (len(self._disc_keys)+ len(self._if_keys)))
                + 'and {:.2f} full density evaluations.'.format(n_feval_per_itr))
        if self._disc_keys is None and self._if_keys is not None:
            print('Each iteration of DHMC on average required '
                + '{:.2f} conditional density evaluations per if parameter '.format(n_fupdate_per_itr / len(self._if_keys))
                + 'and {:.2f} full density evaluations.'.format(n_feval_per_itr))
        if self._disc_keys is not None and self._if_keys is None:
            print('Each iteration of DHMC on average required '
                + '{:.2f} conditional density evaluations per discrete parameter '.format(n_fupdate_per_itr / len(self._disc_keys))
                + 'and {:.2f} full density evaluations.'.format(n_feval_per_itr))

        print(50*'=')
        print('Sampling has now been completed....')
        print(50*'=')
        all_samples = pd.DataFrame.from_dict(x_dicts, orient='columns', dtype=float)
        all_samples = all_samples[self._state.all_vars]
        all_samples.rename(columns=self._names, inplace=True)
        # here, names.values() are the true keys
        samples = copy.copy(all_samples.loc[burn_in:])
        # WORKs REGARDLESS OF type of params (i.e np.arrays, variables, torch.tensors, floats etc) and size. Use samples['param_name'] to extract
        # all the samples for a given parameter

        stats = {'samples': samples, 'samples_wo_burin': all_samples,
                 'stats': extract_stats(samples, keys=list(self._names.values())),
                 'stats_wo_burnin': extract_stats(all_samples, keys=list(self._names.values())),
                 'accept_rate': np.sum(accept[burn_in:]) / len(accept[burn_in:]),
                 'number_of_function_evals': n_feval_per_itr,
                 'time_elapsed': time_elapsed, 'param_names': list(self._names.values())}

        if print_stats:
            print(stats['stats'])
            print('The acceptance ratio is: {0}'.format(stats['accept_rate']))
        if save_samples:
            save_data(stats['samples'], stats['samples_wo_burin'], prefix='bhmc_chain_{}_'.format(chain_num))
        if plot:
            self.create_plots(stats['samples'], keys=stats['param_names'],lag=lag, burn_in=plot_burnin, ac=plot_ac)
        if plot_graphmodel:
            self.model_graph.display_graph()
        return stats  # dict

    def create_plots(self, dataframe_samples, keys, lag, all_on_one=True, save_data=False, burn_in=False, ac=False):
        """

        :param keys:
        :param ac: bool
        :return: Generates plots
        """

        plot_object = plot(dataframe_samples=dataframe_samples, keys=keys, lag=lag, burn_in=burn_in)
        plot_object.plot_density(all_on_one)
        plot_object.plot_trace(all_on_one)
        if ac:
            plot_object.auto_corr()

    def sample_multiple_chains(self, n_chains=1, n_samples=1000, burn_in=1000, stepsize_range=[0.05, 0.20],
                               n_step_range=[5, 20], seed=None, n_update=10, lag=20,
                               print_stats=False, plot=False, plot_graphmodel=False, save_samples=False,
                               plot_burnin=False, plot_ac=False):
        '''

        :param n_chains: number of chains to run

        :return: all_stats: dict of multiple chains, each chain contains one stats; stats the return of method sample
               eg. all_stats = {'0': stats}, where stats{'samples': df}
        '''
        all_stats = {}
        for i in range(n_chains):
            all_stats[i] = self.sample(i, n_samples, burn_in, stepsize_range, n_step_range, seed, n_update, lag,
                                       print_stats, plot, plot_graphmodel, save_samples, plot_burnin, plot_ac)

        return all_stats
