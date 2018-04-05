#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:43
Date created:  19/03/2018

License: MIT
'''

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
from pyfo.inference.mcmc import MCMC


class HMC(MCMC):
    '''
    Built on top the inference class. This is the base class for all HMC variants and implements the original
    HMC algorithm .

    References:

    [1] Hybrid Monte Carlo Duane et. al 1987  https://www.sciencedirect.com/science/article/pii/037026938791197X
    [2] `MCMC Using Hamiltonian Dynamics`,Radford M. Neal, 2011

    :param step_size:
    :param step_range:
    :param num_steps:
    :param adapt_step_size:
    :param transforms: Optional dictionary that specifies a transform for a latent variable with constrained supporr.
    The Transform must be invertible and implement `log_abs_det_jacobian'. If None, and latent variables with
    constrained support exist then the inference engine automatically takes advantage of torch.distribtuions.transforms
    to do the transformation automatically.
    :return:
    '''
    def __init__(self, step_size=None, step_range= None, num_steps=None, adapt_step_size=False, transforms=None):

       self.step_size = step_size if step_size is not None else 2
       if step_range is not None:
           self.step_range = step_range
       elif num_steps is not None:
           self.step_range = self.step_size * num_steps
       else:
           self.step_range = 2* math.pi
        self.num_steps = max(1, int(self.num_steps / self.step_size))

       self.adapt_step_size = adapt_step_size
       self._target_accept_prob = 0.8

       self.transforms = {} if transforms is not None else False
       self.automatic_transformed_enbaled = True if transforms is None else False
       super(HMC,self).__init__()


    def _energy(self, x, p, cont_keys):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param x:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :param cont keys: list of continous keys within state
        :return: Tensor
        """
        kinetic_cont = 0.5 * torch.sum(torch.stack([torch.dot(p[name], p[name]) for name in p]))
        kinetic_energy = kinetic_cont
        potential_energy = -self.log_posterior(x)

        return self._state._return_tensor(kinetic_energy) + self._state._return_tensor(potential_energy)

    def sample(self):
        '''

        :return:
        '''
        return 0

    def gauss_laplace_leapfrog(self, x0, p0, stepsize):
        """
        Performs the full DHMC update step. It updates the continous parameters using
        the standard integrator and the discrete parameters via the coordinate wie integrator.

        :param x: Type dictionary
        :param p: Type dictionary
        :param stepsize:
        :param log_grad:
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
        logp = self.log_posterior(x, set_leafs=True)
        for key in self._cont_keys:
            p[key] = p[key] + 0.5*stepsize*self.grad_logp(logp,x[key])


        for key in self._cont_keys:
            x[key] = x[key] + stepsize*self.M * p[key] # full step for postions
        logp = self.log_posterior(x, set_leafs=True)
        for key in self._cont_keys:
            p[key] = p[key] + 0.5*stepsize*self.grad_logp(logp,x[key])
        return x, p, n_feval, n_fupdate, 0


