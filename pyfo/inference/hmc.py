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
       self._reset()
       super(HMC,self).__init__()


    def _energy(self, x, p, cont_keys):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param x:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :param cont keys: list of continous keys within state
        :return: Tensor
        """
        if self._disc_keys is not None:
            kinetic_disc = torch.sum(
                torch.stack([self.M * torch.dot(torch.abs(p[name]), torch.abs(p[name])) for name in self._disc_keys]))
        else:
            kinetic_disc = VariableCast(0)
        if self._cont_keys is not None:
            kinetic_cont = 0.5 * torch.sum(torch.stack([torch.dot(p[name], p[name]) for name in self._cont_keys]))
        else:
            kinetic_cont = VariableCast(0)
        if self._if_keys is not None:
            kinetic_if = torch.sum(
                torch.stack([self.M * torch.dot(torch.abs(p[name]), torch.abs(p[name])) for name in self._if_keys]))
        else:
            kinetic_if = VariableCast(0)
        kinetic_energy = kinetic_cont + kinetic_disc + kinetic_if
        potential_energy = -self.log_posterior(x)

        return self._state._return_tensor(kinetic_energy) + self._state._return_tensor(potential_energy)

    def sample(self):
        '''

        :return:
        '''

