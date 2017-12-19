#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:08
Date created:  08/12/2017

License: MIT
'''

import torch
import importlib
import os
import math
from torch.autograd import Variable
import numpy as np
import sys
from itertools import permutations
import time
from pyfo.utils.core import VariableCast
import pandas as pd
# the state interacts with the interface, where ever that is placed....
from pyfo.models import state

# def load_from_file(filepath):
#     '''
#     Helper function for extracting the whole module and not just the package.
#     See answer by clint miller for details:
#     https://stackoverflow.com/questions/301134/dynamic-module-import-in-python?noredirect=1&lq=1
#     :param file_path
#     :type string
#     :return module of model (models base class is the interface)
#     '''
#     PATH = '././FOPPL/outbook-pytorch/' + model_name +'.py'
#     mod = __import__(model_name)
#     components = mod_name.split('.')
#     for comp in components[1:]:
#         mod = getattr(mod, comp)
#     return mod

class DHMCSampler(object):
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
        ## momentum drawn from the laplacian, and the continous params to be drawn from
        ## gaussian
        #4 TH NOTE
        ## Need to deal with a M matrix. I may  just set it to 1 everywhere, inferring the identity.

        if scale is None:
             scale = VariableCast(torch.ones(chains)) # outputs chains x n_param tensor

        # # Set the scale of v to be inversely proportional to the scale of x.

        self.model =cls() # instantiates model
        self._state =state(cls)
        self._disc_keys = self._state._return_disc_list()
        self._cont_keys = self._state._return_cont_list()
        self.grad_logp = self._state._grad_logp
        self.init_state = self._state.intiate_state() # essentially this is just x0
        self.n_disc = len(self._disc_keys)
        self.n_cont = len(self._cont_keys)
        self.n_params =  self.n_disc + self.n_cont
        # Note:
        # Need to redefine the M matrix for dictionaries.
        #self.M = torch.div(VariableCast(1),torch.cat((scale[:,:-n_disc]**2, scale[:,-n_disc:]),dim=1)) # M.size() = (chains x n_param)
        self.M = 1 #Assumes the identity matrix and assumes everything is 1d for now.

        self.log_posterior = state.log_posterior  # returns a function that thats the current state as argument
        self.logp_update = state.logp_update # returns a function ; not currently used 10:50 14th Dec.
        # if M is None:
        #     self.M = torch.ones(n_param, 1) # This M.size() = (10,1)

    def random_momentum(self):
        """
        Constructs a momentum dictionary where for the discrete keys we have laplacian momen tum
        and for continous keys we have gaussian
        :return:
        """
        p = {}
        if self._disc_keys is not None:
            for key in self._disc_keys:
                p[key] = self.M[key] * VariableCast(np.random.laplace(size=1)) #in the future add support for multiple dims
        if self._cont_keys is not None:
            for key in self._cont_keys:
                p[key] = VariableCast(torch.sqrt(self.M[key]) * torch.normal(0,1))  # in the future make for multiple dims
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

        logp_diff = self.log_posterior(x_star, set_leafs=False) - self.log_posterior(x, set_leafs=False)

        if torch.gt(self.M*torch.abs(torch.sign(p[key])),logp_diff)[0][0]:
            x = x_star
            p[key] = p[key] + torch.sign(p[key])*self.M*logp_diff
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
        :return: x, p the proposed values as dict.
        """

        # number of function evaluations and fupdates for discrete parameters
        n_feval = 0
        n_fupdate = 0

        #performs shallow copy
        x = x0.copy.copy()
        p = p0.copy.copy()

        # perform first step of leapfrog integrators
        if self._cont_keys is not None:
            logp = self.log_posterior(x, set_leafs=True)
            for key in self._cont_keys:
                p[key] = p[key] + 0.5*stepsize*self.grad_logp(logp,x[key]) # Need to make sure this works

        if self._disc_keys is None:
            for key in self._cont_keys:
                x[key] = x[key] + stepsize*self.M * p[key] # full step for postions

        else:
            permuted_keys = permutations(self._disc_keys,1)
            # permutates all keys into one permutated config. It deletes in memory as each key is called
            # returns a tuple ('key', []), hence to call 'key' requires [0] index.

            if self._cont_keys is not None:
                for key in self._cont_keys:
                    x[key] = x[key] + 0.5 * stepsize * self.M * p[key]
                logp = self.log_posterior(x, set_leafs=True)


            if math.isinf(logp):
                return x, p, n_feval, n_fupdate

            if self._disc_keys is not None:
                print('update coordinate wise')
                for key in permuted_keys:
                    x[key[0]], p[key[0]] = self.coordInt(x, p, stepsize, key[0])
                n_fupdate += 1

            if self._cont_keys is not None:
                for key in self._cont_keys:
                    x[key] = x[key] + 0.5 * stepsize * self.M * p[key] # final half step for position

            if not self._cont_keys:
                logp = self.log_posterior(x, set_leafs=True)
                for key in self._cont_keys:
                    p[key] = p[key] + 0.5*stepsize*self._grad_logp(logp, x[key]) # final half step for momentum
                n_feval += 1
            return x, p, n_feval, n_fupdate


    def _energy(self, x, p):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param x:
        :param p:
        :return: Tensor
        """

        kinetic_cont = 0.5 * torch.sum(torch.stack([self.M*p[name]**2 for name in self._cont_keys]))
        kinetic_disc = torch.sum(torch.stack([self.M*torch.abs(p[name]) for name in self._disc_keys]))
        kinetic_energy = kinetic_cont + kinetic_disc
        potential_energy = -self._log_prob(x)


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

    def sample(self,n_samples= 1000, burn_in= 1000, stepsize_range = [0.05,0.20], n_step_range=[5,20],seed=None, n_update=10):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x = self.init_state
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
        # WORKs REGARDLESS OF type of params and size. Use samples['param_name'] to extract
        # all the samples for a given parameter
        stats = {'samples':samples, 'accept_prob': accept_prob, 'number_of_function_evals':n_feval_per_itr, \
                 'time_elapsed':time_elapsed}

        return stats



