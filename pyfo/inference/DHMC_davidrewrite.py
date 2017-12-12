#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:08
Date created:  08/12/2017

License: MIT
'''

import torch
import copy
import math
from torch.autograd import Variable
import numpy as np
import sys
from itertools import permutations
import time
from pyfo.utils.core import VariableCast
import pandas as pd
class DHMCSampler(object):

    def __init__(self, state, n_param, chains= 1,  scale=None):
        # Note for self:
        ## state is a class that contains a dictionary of the system.
        ## to get log_posterior and log_update call the methods on the
        ## state
        # 2nd Note:
        ## Rather than dealing with the 'indicies' of parameters we deal instead with the keys
        # 3rd Note
        ## When generating the mometum dictionary we need the discrete params to have their
        ## momentum drawn from the laplacian, and the continous params to be drawn from
        ## gaussian
        if scale is None:
             scale = VariableCast(torch.ones(chains,n_param)) # outputs chains x n_param tensor

        # # Set the scale of v to be inversely proportional to the scale of x.
        self._disc_keys = state._return_disc_list()
        self._cont_keys = state._return_cont_list()
        self.grad_logp = state._grad_logp

        self.n_param = n_param
        self.n_disc = len(self._disc_keys)
        self.n_cont = len(self._cont_keys)
        # Note:
        # Need to redefine the M matrix for dictionaries.
        self.M = torch.div(VariableCast(1),torch.cat((scale[:,:-n_disc]**2, scale[:,-n_disc:]),dim=1)) # M.size() = (chains x n_param)
        self.log_posterior = state.log_posterior  # returns a function that thats the current state as argument
        self.logp_update = state.logp_update # returns a function
        # if M is None:
        #     self.M = torch.ones(n_param, 1) # This M.size() = (10,1)

    def random_momentum(self):
        """
        Constructs a momentum dictionary
        :return:
        """
        p = {}
        if self._disc_keys is not None:
            for key in self._disc_keys:
                p[key] = self.M[key] * VariableCast(np.random.laplace(size=1)) #in the future add support for multiple dims
        if self._cont_keys is not None:
            for key in self._cont_keys:
                p[key] = torch.sqrt(self.M[key]) * torch.normal(0,1)  # in the future make for multiple dims
    def coordInt(self,x,p, stepsize,key):
        """
        Performs the coordinate wise update. The permutation is done before the
        variables are executed within the function.

        :param x: Dict of state of positions
        :param p: Dict of state of momentum
        :param stepsize: Float
        :param key: unique parameter String
        :return: Updated x and p indicies
        """

        x_star = x.copy.copy()
        x_star[key] = x_star[key] + stepsize*self.M # Need to change M here again

        logp_diff = self.log_posterior(x_star) - self.log_posterior(x)
        
        if torch.gt(self.M[key]*torch.abs(torch.sign(p[key])),logp_diff)[0][0]:
            x = x_star
            p[key] = p[key] + torch.sign(p[key])*self.M[key]*logp_diff
        else:
            p[key] = -p[key]
        return x,p



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
        :return:
        """

        # number of function evaluations and fupdates for discrete parameters
        n_feval = 0
        n_fupdate = 0

        x = x0.copy.copy()
        p = p0.copy.copy()

        # update continous set of parameters
        for key in self._cont_keys:
            p[key] = p[key] + 0.5*stepsize*self.grad_logp(p[key]) # Need to make sure this works
            # x[key] = x[key] + 0.5*stepsize*self.M * p[key] # This M will not work in current form

        if self._disc_keys is not None:
            permuted_keys = permutations(self._disc_keys,1)
            # permutates all keys into one permutated config. It deletes in memory as each key is called
            # returns a tuple ('key', []), hence to call 'key' requires [0] index.



        if not self._disc_keys:
            print('update coordinate wise')
            for key in permuted_keys:
                x[key[0]], p[key[0]] = self.coordInt(x, p, stepsize, key[0])
            n_fupdate += 1
        if not self._cont_keys:
            # performs standard HMC update
            grad = self._grad_potential(x)
            for key in self._cont_keys:
                x[key] = x[key] + 0.5*stepsize*self.M*p[key]
                p[key] = p[key] + 0.5*stepsize*self._grad_potential(key)
            n_feval += 1
        return x, p, n_feval, n_fupdate

    def _grad_potential(self, x):
        log_joint_prob = self._log_prob(x)
        log_joint_prob.backward()
        grad_potential = {}
        for name, value in x.items():
            grad_potential[name] = -value.grad.clone().detach()
            grad_potential[name].volatile = False
        return grad_potential

    def _energy(self, x, p):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param x:
        :param p:
        :return: Tensor
        """
        kinetic_energy = 0.5 * torch.sum(torch.stack([p[name]**2 for name in p]))
        potential_energy = -self._log_prob(x)


        return self._return_tensor(kinetic_energy) + self._return_tensor(potential_energy)

    @staticmethod
    def _return_tensor(val1):
        """
        Takes a values, checks to see if it is a variable, if so returns tensor,
        else returns a tensor of the value
        :param val1:
        :return:
        """
        if isinstance(val1, Variable):
            return val1.data
        else:
            return val1

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
        intial_energy = -self._energy(x0,p)
        n_feval = 0
        n_fupdate = 0
        x, p, n_feval_local, n_fupdate_local = self.gauss_laplace_leapfrog(x0,p,stepsize)
        for i in range(1,n_step):
            # may have to add inf statement see original code
            x,p, n_feval_local, n_fupdate_local = self.gauss_laplace_leapfrog(x,p,stepsize)
            n_feval += n_feval_local
            n_fupdate += n_fupdate_local


        final_energy = -self._energy(x,p)
        acceptprob  = torch.min(torch.ones(1),torch.exp(final_energy - intial_energy)) # Tensor

        if acceptprob[0] < np.random.uniform(0,1):
            x = x0

        return x, acceptprob, n_feval, n_fupdate

    def run_sampler(self,n_samples= 1000, burn_in= 1000, stepsize_range = [0.05,0.20], n_step_range=[5,20], x0=None,,seed=None, n_update=10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if x0 is None:
            sys.exit('Warning no latent parameters specified. Program terminated')
        x = x0
        n_per_update = math.ceil((n_samples + burn_in)/n_update)
        n_feval = 0
        n_fupdate = 0
        rows = []
        accept_prob = torch.zeros(n_samples + burn_in, 1)

        tic = time.process_time()

        for i in range(n_samples+burn_in):
            stepsize = np.random.uniform_(stepsize_range[0], stepsize_range[1]) #  may need to transforms to variables.
            n_step = np.random.uniform_(n_step_range[0], n_step_range[1])
            x, accept_prob[i], n_feval_local, n_fupdate_local = self.hmc(stepsize,n_step,x)
            n_feval += n_feval_local
            n_fupdate += n_fupdate_local
            rows.append(x)
            if (i + 1) % n_per_update == 0:
                print('{:d} iterations have been completed.'.format(i + 1))
        toc = time.process_time()
        time_elapsed = toc - tic
        n_feval_per_itr = n_feval / (n_samples + burn_in)
        n_fupdate_per_itr = n_fupdate / (n_samples + burn_in)
        print('Each iteration of DHMC on average required '
            + '{:.2f} conditional density evaluations per discontinuous parameter '.format(n_fupdate_per_itr / len(self._disc_keys))
            + 'and {:.2f} full density evaluations.'.format(n_feval_per_itr))

        samples = pd.DataFrame.from_dict(rows, orient='columns')
        return samples, accept_prob, n_feval_per_itr, time_elapsed# [total_accept, burnin_accept]



