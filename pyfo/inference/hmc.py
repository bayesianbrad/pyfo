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
import torch.distributions as dists
from torch.autograd import Variable
from torch.distributions import constraints, biject_to
import pyfo.distributions as dist


# the state interacts with the interface, where ever that is placed....
from pyfo.utils import state
from pyfo.utils.core import VariableCast
from pyfo.utils.eval_stats import extract_stats
from pyfo.utils.eval_stats import save_data
from pyfo.utils.plotting import Plotting as plot
from pyfo.inference.mcmc import MCMC
from pyfo.utils.core import DualAveraging

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
    def __init__(self, step_size=None,  num_steps=None, adapt_step_size=False, trajectory_length=None, **kwargs):


       self.step_size = step_size if step_size is not None else 2
       if trajectory_length is not None:
           self.trajectory_length = trajectory_length
       elif num_steps is not None:
           self.trajectory_length = self.step_size * num_steps
       else:
           self.trajectory_length = 2 * math.pi  # from Stan
       self.num_steps = max(1, int(self.trajectory_length / self.step_size))

       self.adapt_step_size = adapt_step_size
       self._target_accept_prob = 0.8

       self.kwargs = kwargs # see pyfo.inference.mcmc.run_inference() for more details




       super(HMC,self).__init__()

    def _find_reasonable_step_size(self, state):
        step_size = self.step_size
        # NOTE: This target_accept_prob is 0.5 in NUTS paper, is 0.8 in Stan,
        # and is different to the target_accept_prob for Dual Averaging scheme.
        # We need to discuss which one is better.
        target_accept_logprob = math.log(self._target_accept_prob)

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        p = self.momentum_sample()
        energy_current = self._energy(state, p)
        state_new, p_new, potential_energy = self.leapfrog_step(state, p, step_size)
        energy_new = potential_energy + self._kinetic_energy(p_new)
        delta_energy = energy_new - energy_current
        # direction=1 means keep increasing step_size, otherwise decreasing step_size
        direction = 1 if target_accept_logprob < -delta_energy else -1

        # define scale for step_size: 2 for increasing, 1/2 for decreasing
        step_size_scale = 2 ** direction
        direction_new = direction
        # keep scale step_size until accept_prob crosses its target
        # TODO: make thresholds for too small step_size or too large step_size
        while direction_new == direction:
            step_size = step_size_scale * step_size
            state_new, p_new, potential_energy = self._leapfrog_step(
                state, p, self._potential_energy(state), step_size)
            energy_new = potential_energy + self._kinetic_energy(p_new)
            delta_energy = energy_new - energy_current
            direction_new = 1 if target_accept_logprob < -delta_energy else -1
        return step_size

    def _adapt_step_size(self, accept_prob):
        # calculate a statistic for Dual Averaging scheme
        H = self._target_accept_prob - accept_prob
        self._adapted_scheme.step(H)
        log_step_size, _ = self._adapted_scheme.get_state()
        self.step_size = math.exp(log_step_size)
        self.num_steps = max(1, int(self.trajectory_length / self.step_size))

    def setup(self, state):

        for name in self._cont_latents:
            p_loc = torch.zeros(state[name].size())
            p_scale = torch.ones(state[name].size())
            self._p[name] = dist.Normal(loc=p_loc, scale=p_scale)
            if self.transforms[name].support is not constraints.real and self._automatic_transform_enabled:
                self.transforms[name] = biject_to(state[name].support).inv #TODO: Go over this with a fine comb.

        if self.adapt_step_size:
            for name, transform in self.transforms.items():
                state[name] = transform(state[name])
            self.step_size = self._find_reasonable_step_size(state)
            self.num_steps = max(1, int(self.trajectory_length / self.step_size))
            # make prox-center for Dual Averaging scheme
            loc = math.log(10 * self.step_size)
            self._adapted_scheme = DualAveraging(prox_center=loc)

    def end_warmup(self):
        if self.adapt_step_size:
            self.adapt_step_size = False
            _, log_step_size_avg = self._adapted_scheme.get_state()
            self.step_size = math.exp(log_step_size_avg)
            self.num_steps = max(1, int(self.trajectory_length / self.step_size))
    def _kinetic(self, p):
        """

        :param p: type: torch.tensor descrip: momentum

        :return: scalar of kinetic energy
        TODO Implement for batching chains
        """

        return 0.5 * torch.sum(torch.stack([torch.dot(p[key], p[key]) for key in self._cont_latents]))

    def _potential_energy(self, state, set_leafs=False):
        state_constrained = state.copy()

        for key, transform in self.transforms.items():
            state_constrained[key] = transform.inv(state_constrained[key])
        potential_energy = -self.__generate_log_pdf(state_constrained, set_leafs=set_leafs)
        # adjust by the jacobian for this transformation.
        for key, transform in self.transforms.items():
            potential_energy = potential_energy + transform.log_abs_det_jacobian(state_constrained[key],
                                                                                     state[key]).sum()
        return potential_energy

    def _energy(self, state, p, cont_latents):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param state:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :param cont keys: list of continuous keys within state
        :return: Tensor
        """



        return self._kinetic_energy(p) + self._potential_energy(state)
    def momentum_sample(self):
        """
        Constructs a momentum dictionary for contin
        and for continous keys we have gaussian
        :return:
        """
        p = dict([[key, torch.randn(state[key].size())] for key in self._cont_latents])
        return p

    def sample(self, state, nsamples= 1000, burnin=100, chains=1, **kwargs):
        '''
        :param nsamples type: int descript: Specifies how many samples you would like to generate.
        :param burnin: type: int descript: Specifies how many samples you would like to remove.
        :param chains :type: int descript: Specifies the number of chains.
        :param save_data :type bool descrip: Specifies whether to save data and return data, or just return.
        :param dirname :type: str descrip: Path to a directory, where data can be saved.

        :return:
        '''

        # automatically transform `state` to unconstrained space, if needed.

        for key, transform in self.transforms.items():
            state[key] = transform(state[key])
        p0 = self.momentum_sample()
        state_new, p_new = self._leapfrog_step(state, p0,
                                       self.step_size, self.num_steps)
        # apply Metropolis correction.
        energy_proposal = self._energy(state_new, p_new)
        energy_current = self._energy(state, p0)
        delta_energy = energy_proposal - energy_current
        rand = np.random.uniform(0,1)
        if rand < (-delta_energy).exp():
            self._accept_cnt += 1
            state = state_new

        if self.adapt_step_size:
            accept_prob = (-delta_energy).exp().clamp(max=1).item()
            self._adapt_step_size(accept_prob)

        self._t += 1

        # get trace with the constrained values for `state`.
        for key, transform in self.transforms.items():
            state[key] = transform.inv(state[key])
        return state

        # note must return pd.DataFrame[self._all_vars].values
        return 0

    def _leapfrog_step(self, state, p, stepsize):
        """
        Performs the full DHMC update step. It updates the continous parameters using
        the standard integrator and the discrete parameters via the coordinate wie integrator.

        :param x: type: dictionary descript: represents the state of the system (all the latents variables and observable
        quantities.
        :param p: type: dictionary descript: represents the momentum for each latent variable
        :param stepsize: type: float
        :return: x, p the proposed values as dict.
        """

        # number of function evaluations and fupdates for discrete parameters
        n_feval = 0
        n_fupdate = 0

        # perform first step of leapfrog integrators
        for i in range(self.num_steps):
            logp = self.__generate_log_pdf(state, set_leafs=True)
            for key in self._cont_latents:
                p[key] = p[key] + 0.5*stepsize*self.__grad_logp(logp,state[key])


            for key in self._cont_latents:
                state[key] = state[key] + stepsize*self.M * p[key] # full step for postions
            logp = self.__generate_pdf(state, set_leafs=True)
            for key in self._cont_latents:
                p[key] = p[key] + 0.5*stepsize*self.____grad_logp(logp,[key])
        return state, p, logp


