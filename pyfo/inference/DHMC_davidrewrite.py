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
from torch.autograd import Variable
import numpy as np
from itertools import permutations
from pyfo.utils.core import VariableCast

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
            p[key] = p[key] + torch.sign(p[key])*M[key]*logp_diff
        else:
            p[key] = -p[key]
        return x,p



    def gauss_laplace_leapfrog(self, x0, p0, stepsize, log_grad, aux, n_disc=0):
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
            x[key] = x[key] + 0.5*stepsize*self.M * p[key] # This M will not work in current form

        if self._disc_keys is not None
            permuted_keys = permutations(self._disc_keys,1)
            # permutates all keys into one permutated config. It deletes in memory as each key is called
            # returns a tuple ('key', []), hence to call 'key' requires [0] index.



        if not self._disc_keys:
            print('update coordinate wise')
            for key in permuted_keys:
                x[key[0]], p[key[0]] = self.coordInt(x, p, stepsize, key[0])
        if not self._cont_keys:
            # performs standard HMC update
            for key in self._cont_keys:
                x[key] = x[key] + 0.5*stepsize*self.M*p[key]
                p[key] = p[key] + 0.5*stepsize*self._grad_potential(x[key])


    def _grad_potential(self, x):
        log_joint_prob = self._log_prob(x)
        log_joint_prob.backward()
        grad_potential = {}
        for name, value in x.items():
            grad_potential[name] = -value.grad.clone().detach()
            grad_potential[name].volatile = False
        return grad_potential

    def _energy(self, x, p):
        kinetic_energy = 0.5 * torch.sum(torch.stack([p[name]**2 for name in p]))
        potential_energy = -self._log_prob(x)
        return kinetic_energy + potential_energy


    def hmc(self, stepsize, n_step, x0,logp0, grad0, aux0):
        """

        :param stepsize:
        :param n_step:
        :param x0:
        :param logp0: probably won't require as will be handled by state
        :param grad0: probably won't require as will be handled by state
        :param aux0:
        :return:
        """

        p = self.random_momentum()



