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
    # def list_to_tensor(self, params):
    #     ''' Unpacks the parameters list tensors and converts it to list
    #
    #     returns tensor of  num_rows = len(values) and num_cols  = 1
    #     problem:
    #         if there are col dimensions greater than 1, then this will not work'''
    #     assert(isinstance(params, list))
    #     temp = Variable(torch.Tensor(len(params)).unsqueeze(-1))
    #     for i in range(len(params)):
    #         temp[i,:] = params[i]
    #     return temp
    def generate_new_step_traj(self):
        ''' Generates a new step adn trajectory size  '''
        step_size = np.random.uniform(self.min_step, self.max_step)
        traj_size = int(np.random.uniform(self.min_traj, self.max_traj))
        return step_size, traj_size


    def leapfrog(self, p_init, values, grad_init):
        '''Performs the leapfrog steps of the HMC for the specified trajectory
        length, given by num_steps
        Parameters
        ----------
            values_init
            p_init
            grad_init     - Description: contains the initial gradients of the joint w.r.t parameters.

        Outputs
        -------
            values -    Description: proposed new values
            p      -    Description: proposed new auxillary momentum
        '''
        step_size, traj_size = self.generate_new_step_traj()
        values_init = values
        self.kinetic = Kinetic(p_init)
        # Start by updating the momentum a half-step and values by a full step
        p = p_init + 0.5 * step_size * grad_init
        values = values_init + step_size * self.kinetic.gauss_ke(p, grad=True)
        for i in range(traj_size - 1):
            # range equiv to [2:nsteps] as we have already performed the first step
            # update momentum
            p = p + step_size * self.potential.eval(values, grad=True)
            # update values
            values = values + step_size * self.kinetic.gauss_ke(p, grad=True)

        # Do a final update of the momentum for a half step
        p = p + 0.5 * step_size * self.potential.eval(values, grad=True)
        # print('Debug p, vlaues leapfrog', p, values)

        # return new proposal state
        return values, p

    # # def discIntegrator(self, values, p_init, ):
    # # def coord_wise(self,values, p_init, i):
    #     '''
    #     Discountinous HMC Nishimura et al. Algorithm 1.
    #     The p here has to have been sampled from a Laplace distribution
    #
    #      '''

