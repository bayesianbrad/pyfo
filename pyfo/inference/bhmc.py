#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:08
Date created:  08/12/2017

License: MIT
'''

import math
import time
from itertools import permutations

import numpy as np
import pandas as pd
import torch
import copy
# the state interacts with the interface, where ever that is placed....
from pyfo.utils import state
from pyfo.utils.core import VariableCast
from pyfo.utils.eval_stats import extract_stats
from pyfo.utils.eval_stats import save_data
from pyfo.utils.plotting import Plotting as plot
#TODO Complete this integrator
class BHMCSampler(object):
    """
    In general model will be the output of the foppl compiler, it is not entirely obvious yet where this
    will be stored. But for now, we will inherit the model from pyro.models.<model_name>
    """

    def __init__(self, cls, chains=1,  scale=None):

        # Note for self:
        ## state is a class that contains a dictionary of the system.
        ## to get log_posterior and log_update call the methods on the
        ## state
        # 2nd Note:
        ## Rather than dealing with the 'indicies' of parameters we deal instead with the keys
        # 3rd Note
        ## When generating the mometum dictionary we need the discrete params to have their
        ## momentum drawn from the laplacian, and the continuous params to be drawn from
        ## gaussian
        #4 TH NOTE
        ## Need to deal with a M matrix. I may  just set it to 1 everywhere, inferring the identity.

        self.model =cls() # instantiates model
        self._state = state.State(cls)

        # Parameter keys
        self._disc_keys = self._state._return_disc_list()
        self._cont_keys = self._state._return_cont_list()
        self._cond_keys = self._state._return_cond_list()
        self._if_keys = self._state._return_if_list()
        self._all_keys = self._state._return_all_list()

        # True latent variable names
        self._names = self._state._return_true_names()

        self.grad_logp = self._state._grad_logp
        self.init_state = self._state.intiate_state() # this is just x0
        # self.n_disc = len(self._disc_keys)
        # self.n_cont = len(self._cont_keys)
        # self.n_params =  self.n_disc + self.n_cont
        self.M = 1 #Assumes the identity matrix and assumes everything is 1d for now.

        self.log_posterior = self._state._log_pdf  # returns a function that is the current state as argument
        # self.logp_update = self._state._log_pdf_update # returns a function ; not currently used 10:50 14th Dec.

    def random_momentum(self):
        """
        Constructs a momentum dictionary where for the discrete keys we have laplacian momen tum
        and for continous keys we have gaussian
        :return:
        """
        p = {}
        if self._disc_keys is not None:
            for key in self._disc_keys:
                p[key] = self.M * VariableCast(np.random.laplace(size=1)) #in the future add support for multiple dims
        if self._cont_keys is not None:
            for key in self._cont_keys:
                p[key] = VariableCast(self.M * torch.normal(torch.FloatTensor([0]),torch.FloatTensor([1])))
                # TODO in the future make for multiple dims
        if self._if_keys is not None:
            for key in self._if_keys:
                p[key] = self.M * VariableCast(np.random.laplace(size=1))
        return p

    def coordInt(self,x,p,stepsize,key):
        """
        Performs the coordinate wise update. The permutation is done before the
        variables are executed within the function.

        :param x: Dict of state of positions
        :param p: Dict of state of momentum
        :param stepsize: Float
        :param key: unique parameter String
        :return: Updated x and p indicies
        """

        x_star = copy.copy(x)
        x_star[key] = x_star[key] + stepsize*self.M*torch.sign(p[key]) # Need to change M here again
        logp_diff = self.log_posterior(x_star, set_leafs=False) - self.log_posterior(x, set_leafs=False)
        cond = torch.gt(self.M*torch.abs(p[key]),-logp_diff)
        if cond.data[0]:
            p[key] = p[key] + torch.sign(p[key])*self.M*logp_diff
            return x_star[key], p[key]
        else:
            p[key] = -p[key]
            return x[key],p[key]

    def gauss_laplace_leapfrog(self, x0, p0, stepsize, aux= None):
        """
        Performs the full DHMC update step. It updates the continous parameters using
        the standard integrator and the discrete parameters via the coordinate wie integrator.

        :param x: Type dictionary
        :param p: Type dictionary
        :param stepsize:
        :param log_grad:
        :param aux:
        :param n_disc:
        :return: x, p the proposed values as dict.
        """

        # number of function evaluations and fupdates for discrete parameters
        n_feval = 0
        n_fupdate = 0

        #performs shallow copy
        x = copy.copy(x0)
        p = copy.copy(p0)
        # perform first step of leapfrog integrators
        if self._cont_keys is not None:
            logp = self.log_posterior(x, set_leafs=True)
            for key in self._cont_keys:
                p[key] = p[key] + 0.5*stepsize*self.grad_logp(logp,x[key])

        if self._disc_keys is None and self._if_keys is None:
            for key in self._cont_keys:
                x[key] = x[key] + stepsize*self.M * p[key] # full step for postions
            logp = self.log_posterior(x, set_leafs=True)
            for key in self._cont_keys:
                p[key] = p[key] + 0.5*stepsize*self.grad_logp(logp,x[key])
            return x, p, n_feval, n_fupdate

        else:
            permuted_keys_list = []
            if self._disc_keys is not None:
                permuted_keys_list = permuted_keys_list + self._disc_keys
            if self._if_keys is not None:
                permuted_keys_list = permuted_keys_list + self._if_keys
            permuted_keys = permutations(permuted_keys_list,1)
            # permutates all keys into one permutated config. It deletes in memory as each key is called
            # returns a tuple ('key', []), hence to call 'key' requires [0] index.

            if self._cont_keys is not None:
                for key in self._cont_keys:
                    x[key] = x[key] + 0.5 * stepsize * self.M * p[key]
                logp = self.log_posterior(x, set_leafs=True)

            # Uncomment this statement when correct transforms have
            # been impleemnted.
            # if math.isinf(logp):
            #     return x, p, n_feval, n_fupdate

            for key in permuted_keys:
                x[key[0]], p[key[0]] = self.coordInt(x, p, stepsize, key[0])
            n_fupdate += 1

            if self._cont_keys is not None:
                for key in self._cont_keys:
                    x[key] = x[key] + 0.5 * stepsize * self.M * p[key] # final half step for position

            if self._cont_keys is not None:
                logp = self.log_posterior(x, set_leafs=True)
                for key in self._cont_keys:
                    p[key] = p[key] + 0.5*stepsize*self.grad_logp(logp, x[key]) # final half step for momentum
                n_feval += 1
            return x, p, n_feval, n_fupdate

    def _energy(self, x, p):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param x:
        :param p:
        :return: Tensor
        """
        if self._disc_keys is not None:
            kinetic_disc = torch.sum(torch.stack([self.M * torch.abs(p[name]) for name in self._disc_keys]))
        else:
            kinetic_disc = VariableCast(0)
        if self._cont_keys is not None:
            kinetic_cont = 0.5 * torch.sum(torch.stack([self.M*p[name]**2 for name in self._cont_keys]))
        else:
            kinetic_cont = VariableCast(0)
        if self._if_keys is not None:
            kinetic_if =  torch.sum(torch.stack([self.M * torch.abs(p[name]) for name in self._if_keys]))
        else:
            kinetic_if = VariableCast(0)
        kinetic_energy = kinetic_cont + kinetic_disc + kinetic_if
        potential_energy = -self.log_posterior(x)

        return self._state._return_tensor(kinetic_energy) + self._state._return_tensor(potential_energy)

    def hmc(self, stepsize, n_step, x0):
        """

        :param stepsize_range: List
        :param n_step_range: List
        :param x0:
        :param logp0: probably won't require as will be handled by state self.log_posterior
        :param grad0: probably won't require as will be handled by state self._grad_log
        :param aux0:
        :return:
        """

        p = self.random_momentum()
        intial_energy = self._energy(x0,p)
        n_feval = 0
        n_fupdate = 0
        x, p, n_feval_local, n_fupdate_local = self.gauss_laplace_leapfrog(x0,p,stepsize)
        for i in range(n_step-1):
            # may have to add inf statement see original code
            x,p, n_feval_local, n_fupdate_local = self.gauss_laplace_leapfrog(x,p,stepsize)
            n_feval += n_feval_local
            n_fupdate += n_fupdate_local


        final_energy = self._energy(x,p)
        acceptprob  = torch.min(torch.ones(1),torch.exp(final_energy - intial_energy)) # Tensor
        if acceptprob[0] < np.random.uniform(0,1):
            x = x0

        return x, acceptprob[0], n_feval, n_fupdate
    def sample(self,n_samples= 1000, burn_in= 1000, stepsize_range = [0.05,0.20], n_step_range=[5,20],seed=None, n_update=10, lag=20, print_stats=False , plot=False, save_samples=False, plot_burnin=False, plot_ac=False):
        # Note currently not doing anything with burn in

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        else:
            print('seed deactivated')
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
        for i in range(n_samples+burn_in):
            stepsize = VariableCast(np.random.uniform(stepsize_range[0], stepsize_range[1])) #  may need to transforms to variables.
            n_step = np.ceil(np.random.uniform(n_step_range[0], n_step_range[1])).astype(int)
            x, accept_prob, n_feval_local, n_fupdate_local = self.hmc(stepsize,n_step,x)
            n_feval += n_feval_local
            n_fupdate += n_fupdate_local
            accept.append(accept_prob)
            x_numpy = copy.copy(x)
            x_dicts.append(self._state.convert_dict_vars_to_numpy(x_numpy))
            if (i + 1) % n_per_update == 0:
                print('{:d} iterations have been completed.'.format(i + 1))
        toc = time.process_time()
        time_elapsed = toc - tic
        n_feval_per_itr = n_feval / (n_samples + burn_in)
        n_fupdate_per_itr = n_fupdate / (n_samples + burn_in)
        # if self._disc_keys or self._if_keys is not None:
        #     print('Each iteration of DHMC on average required '
        #         + '{:.2f} conditional density evaluations per discontinuous parameter '.format(n_fupdate_per_itr / len(self._disc_keysgit s))
        #         + 'and {:.2f} full density evaluations.'.format(n_feval_per_itr))



        all_samples = pd.DataFrame.from_dict(x_dicts, orient='columns').astype(dtype='float')
        all_samples.rename(columns=self._names, inplace=True)

        samples =  all_samples.loc[burn_in:, :]
        # here, names.values() are the true keys
        print(50*'=')
        print('Sampling has now been completed....')
        print(50*'=')
        # WORKs REGARDLESS OF type of params (i.e np.arrays, variables, torch.tensors, floats etc) and size. Use samples['param_name'] to extract
        # all the samples for a given parameter
        stats = {'samples':samples, 'samples_wo_burin':all_samples, 'stats':extract_stats(samples), 'stats_wo_burnin': extract_stats(all_samples), 'accept_prob': np.sum(accept[burn_in:])/len(accept), 'number_of_function_evals':n_feval_per_itr, \
                 'time_elapsed':time_elapsed, 'param_names': list(self._names.values())}
        if print_stats:
            print(stats['stats'])
        if save_samples:
            save_data(stats['samples'], stats['samples_wo_burin'], stats['param_names'])
        if plot:
            self.create_plots(stats['samples'], stats['samples_wo_burin'], keys=stats['param_names'],lag=lag, burn_in=plot_burnin, ac=plot_ac)

        return stats

    def create_plots(self, dataframe_samples,dataframe_samples_woburin, keys, lag, all_on_one=True, save_data=False, burn_in=False, ac=False):
        """

        :param keys:
        :param ac: bool
        :return: Generates plots
        """

        plot_object = plot(dataframe_samples=dataframe_samples,dataframe_samples_woburin=dataframe_samples_woburin, keys=keys,lag=lag, burn_in=burn_in )
        plot_object.plot_density(all_on_one)
        plot_object.plot_trace(all_on_one)
        if ac:
            plot_object.auto_corr()



