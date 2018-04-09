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
       self.initialize()
       self.generate_latent_vars()

       super(HMC,self).__init__()

    def _kinetic(self, p):
        """

        :param p:
        :return:

        """

        return 0.5 * torch.sum(torch.stack([torch.dot(p[key], p[key]) for key in self._cont_latents]))

    def _potential_energy(self, state, set_leafs=False):
        state_constrained = state.copy()

        for key, transform in self.transforms.items():
            state_constrained[key] = transform.inv(state_constrained[key])
        potential_energy = self.__generate_log_pdf(state_constrained, set_leafs=set_leafs)
        # adjust by the jacobian for this transformation.
        for key, transform in self.transforms.items():
            potential_energy = potential_energy + transform.log_abs_det_jacobian(state_constrained[key],
                                                                                     state[key]).sum()
        return potential_energy

    def _energy(self, state, p, cont_keys):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param state:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :param cont keys: list of continous keys within state
        :return: Tensor
        """

        potential_energy = -self.__potential_energy(x)

        return self._kinetic_energy + potential_energy

    def momentum_sample(self):
        """
        Constructs a momentum dictionary where for the discrete keys we have laplacian momen tum
        and for continous keys we have gaussian
        :return:
        """
        p = {}
        VariableCast(self.M * np.random.randn(self._sample_sizes[key]))
        for key in self._cont_keys:
            p[key] = VariableCast(self.M * np.random.randn(self._sample_sizes[key]))
        return p
    
    def sample(self):
        '''

        :return:
        '''

        # automatically transform `z` to unconstrained space, if needed.
        for key, transform in self.transforms.items():
            state[key] = transform(state[key])
        p = self.momentum_sample()
        state_new, p_new = velocity_verlet(z, r,
                                       self._potential_energy,
                                       self.step_size,
                                       self.num_steps)
        # apply Metropolis correction.
        energy_proposal = self._energy(z_new, r_new)
        energy_current = self._energy(z, r)
        delta_energy = energy_proposal - energy_current
        rand = pyro.sample('rand_t='.format(self._t), dist.Uniform(ng_zeros(1), ng_ones(1)))
        if rand.log() < -delta_energy:
            self._accept_cnt += 1
            z = z_new
        self._t += 1

        # get trace with the constrained values for `z`.
        for key, transform in self.transforms.items():
            z[key] = transform.inv(z[key])
        return self._get_trace(z)
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
        logp = self.__generate_pdf(x, set_leafs=True)
        for key in self._cont_keys:
            p[key] = p[key] + 0.5*stepsize*self.__grad_logp(logp,x[key])


        for key in self._cont_keys:
            x[key] = x[key] + stepsize*self.M * p[key] # full step for postions
        logp = self.__generate_pdf(x, set_leafs=True)
        for key in self._cont_keys:
            p[key] = p[key] + 0.5*stepsize*self.____grad_logp(logp,x[key])
        return x, p, n_feval, n_fupdate, 0


