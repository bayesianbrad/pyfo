#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:49
Date created:  06/09/2017

License: MIT
'''
import torch
import numpy as np
from Utils.kinetic import Kinetic
from torch.autograd import Variable

class Integrator():

    def __init__(self, potential, min_step, max_step, max_traj, min_traj):
        self.potential = potential
        if min_step is None:
            self.min_step = np.random.uniform(0.01, 0.07)
        else:
            self.min_step = min_step
        if max_step is None:
            self.max_step = np.random.uniform(0.07, 0.18)
        else:
            self.max_step = max_step
        if max_traj is None:
            self.max_traj = np.random.uniform(18, 25)
        else:
            self.max_traj = max_traj
        if min_traj is None:
            self.min_traj = np.random.uniform(1, 18)
        else:
            self.min_traj = min_traj
    def generate_new_step_traj(self):
        ''' Generates a new step adn trajectory size  '''
        step_size = np.random.uniform(self.min_step, self.max_step)
        traj_size = int(np.random.uniform(self.min_traj, self.max_traj))
        return step_size, traj_size


    def leapfrog_integrator(self, x, p, grad_log, step_size, num_steps):
        """
        Leapfrog integrator.

        :param x: dictionary of sample site names and their current values
        :param p: dictionary of sample site names and corresponding momenta
        :param grad_logp: function that returns gradient of the potential given x
            for each sample site
        :return: (x_next, p_next) having same types as (x, p)

        Note to self: z ---> x ; r ---> p
        """
        step_size, traj_size = self.generate_new_step_traj()
        values_init = values

        # deep copy the current state - (x, p)
        x_next = {key: val.clone().detach() for key, val in x.items()}
        p_next = {key: val.clone().detach() for key, val in p.items()}
        self.retain_grads(x_next)
        grads = grad_logp(x_next)

        for _ in range(num_steps):
            # detach graph nodes for next iteration
            detach_nodes(x_next)
            detach_nodes(z_next)
            for site_name in x_next:
                # p(n+1/2)
                p_next[site_name] = p_next[site_name] + 0.5 * step_size * (-grads[site_name])
                # x(n+1)
                x_next[site_name] = x_next[site_name] + step_size * p_next[site_name]
            # retain gradients for intermediate nodes in backward step
            retain_grads(x_next)
            grads = grad_potential(x_next)
            for site_name in p_next:
                # r(n+1)
                p_next[site_name] = p_next[site_name] + 0.5 * step_size * (-grads[site_name])
        return x_next, p_next

    def retain_grads(x):
        for value in x.values():
            # XXX: can be removed with PyTorch 0.3
            if value.is_leaf and not value.requires_grad:
                value.requires_grad = True
            value.retain_grad(

    def detach_nodes(x):
        for key, value in x.items():
            x[key] = Variable(value.data, requires_grad=True)
    # # def discIntegrator(self, values, p_init, ):
    # # def coord_wise(self,values, p_init, i):
    #     '''
    #     Discountinous HMC Nishimura et al. Algorithm 1.
    #     The p here has to have been sampled from a Laplace distribution
    #
    #      '''

