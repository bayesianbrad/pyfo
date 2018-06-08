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
import torch.distributions as dists
from torch.distributions import constraints
import distributions as dist

from inference.mcmc import MCMC
from utils.core import DualAveraging, _generate_log_pdf, _grad_logp, _to_leaf

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
    def __init__(self, model_code=None, step_size=None,  num_steps=None, adapt_step_size=True, trajectory_length=None, **kwargs):
        super(HMC, self).__init__()

        self.model_code = model_code
        self._accept_cnt = 0
        self.__dict__.update(kwargs)

        self.generate_latent_vars()
        self.initialize()


    def _kinetic_energy(self, p):
        """

        :param p: type: torch.tensor descrip: momentum

        :return: scalar of kinetic energy
        """
        # print('Debug statement in _kinetic_energy : printing momentum p : {0}'.format(p[self._cont_latents[0]]))

        return 0.5 * torch.sum(torch.stack([torch.mm(p[key].t(), p[key]) for key in self._cont_latents]))

    def _potential_energy(self, state, set_leafs=True):
        state_constrained = state.copy()
        if set_leafs:
            state_constrained = _to_leaf(state=state_constrained,latent_vars=self._all_vars)
        potential_energy = -_generate_log_pdf(model=self.model,state=state_constrained)
        return potential_energy, state_constrained

    def _energy(self, state, p):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param state:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :param cont keys: list of continuous keys within state
        :return: Tensor
        """

        potential_energy, state_constrained = self._potential_energy(state)

        return self._kinetic_energy(p) + potential_energy, potential_energy, state_constrained

    def momentum_sample(self, state):
        """
        Constructs a momentum dictionary for contin
        and for continous keys we have gaussian
        :return:
        """
        p = dict([[key, torch.randn(state[key].size()[0], 1, requires_grad=False)] for key in self._cont_latents])
        return p

    def sample(self, state):
        '''
        :param nsamples type: int descript: Specifies how many samples you would like to generate.
        :param burnin: type: int descript: Specifies how many samples you would like to remove.
        :param chains :type: int descript: Specifies the number of chains.
        :param save_data :type bool descrip: Specifies whether to save data and return data, or just return.
        :param dirname :type: str descrip: Path to a directory, where data can be saved.
        :return:
        '''

        # automatically transform `state` to unconstrained space, if needed
        self.step_size = np.random.uniform(0.05, 0.18)
        self.num_steps = int(np.random.uniform(10, 20))
        p0 = self.momentum_sample(state)
        energy_current,logp, state_constrained = self._energy(state, p0)
        state_new, p_new, logp = self._leapfrog_step(state_constrained, p0, logp)
        # apply Metropolis correction.
        energy_proposal = logp + self._kinetic_energy(p_new)
        # delta_energy = energy_proposal - energy_current
        alpha = torch.min(torch.exp(energy_current - energy_proposal)).detach().numpy()
        p_accept = min(1, alpha)
        if p_accept > np.random.uniform():
            self._accept_cnt += 1
            state_constrained = state_new
        rand = torch.tensor(np.random.uniform(0,1))
        # accept = torch.lt(rand, torch.exp(-delta_energy)).byte().any().item()
        # if accept:
        #     # print('Debug statement in hmc.sample() : \n Printing : state accepted')
        #     self._accept_cnt += 1
        #     state = state_new
        # print('Acceptance : {0}'.format(self._accept_cnt))
        return state_constrained
    def _leapfrog_step(self, state, p, logp):
        """
        Performs the full DHMC update step. It updates the continous parameters using
        the standard integrator and the discrete parameters via the coordinate wie integrator.

        :param state: type: dictionary descript: represents the state of the system (all the latents variables and observable
        quantities.
        :param p: type: dictionary descript: represents the momentum for each latent variable
        :return: x, p the proposed values as dict.
        """
        # Radford neal implementation
        # logp, state = self._potential_energy(state, set_leafs=True)
        grads = _grad_logp(input=logp, parameters=state, latents=self._cont_latents)
        for key in self._cont_latents:
            p[key] = p[key] -  0.5 * self.step_size * grads[key]
        for i in range(self.num_steps):
            #should be able to make this more efficient]
            for key in self._cont_latents:
                state[key] = state[key] + self.step_size * p[key]  # full step for postions
            logp, state = self._potential_energy(state, set_leafs=True)
            grads = _grad_logp(input=logp, parameters=state, latents=self._cont_latents)
            if i == self.num_steps-1:
                break
            else:
                for key in self._cont_latents:
                    p[key] = p[key] - self.step_size * grads[key]
        # final half step for momentum
        for key in self._cont_latents:
            p[key] = p[key] - 0.5 * self.step_size * grads[key]
        return state, p, logp
