#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:47
Date created:  06/09/2017

License: MIT
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:31:26 2017

@author: bradley
"""

import torch
import numpy as np
import time
import math
from torch.autograd import Variable
from Utils.integrator import Integrator
from Utils.metropolis_step import Metropolis
# np.random.seed(12345)
# torch.manual_seed(12345)


class HMCsampler():
    '''
    Notes:  the params from FOPPL graph will have to all be passed to - maybe
    Methods
    -------
    leapfrog_step - preforms the integrator step in HMC
    hamiltonian   - calculates the value of hamiltonian
    acceptance    - calculates the acceptance probability
    run_sampler

    Attributes
    ----------

    '''
    def __init__(self, program, burn_in= 100, n_samples= 1000, M= None,  min_step= None, max_step= None,\
                 min_traj= None, max_traj= None):
        self.burn_in    = burn_in
        self.n_samples  = n_samples
        self.M          = M
        self.potential  = program()
        self.integrator = Integrator(self.potential, min_step, max_step, \
                                     min_traj, max_traj)
        # self.dim        = dim

        # TO DO : Implement a adaptive step size tuning from HMC
        # TO DO : Have a desired target acceptance ratio
        # TO DO : Implement a adaptive trajectory size from HMC


    def run_sampler(self):
        ''' Runs the hmc internally for a number of samples and updates
        our parameters of interest internally
        Parameters
        ----------
        n_samples
        burn_in

        Output
        ----------
        A tensor of the number of required samples
        Acceptance rate
        '''
        print(' The sampler is now running')
        # In the future dim = # of variables will not be needed as Yuan will provide
        # that value in the program and it shall return the required dim.
        logjoint_init, values_init, grad_init, dim = self.potential.generate()
        metropolis   = Metropolis(self.potential, self.integrator, self.M)
        temp,count   = metropolis.acceptance(values_init, logjoint_init, grad_init)
        samples      = Variable(torch.zeros(self.n_samples,dim))
        samples[0]   = temp.data.t()


        # Then run for loop from 2:n_samples
        for i in range(self.n_samples-1):
            logjoint_init, grad_init = self.potential.eval(temp, grad_loop= True)
            temp, count = metropolis.acceptance(temp, logjoint_init, grad_init)
            samples[i + 1, :] = temp.data.t()
            # try:
            #     samples[i+1,:] = temp.data.t()
            # except RuntimeError:
            #     print(i)
            #     break
            # update parameters and draw new momentum
            if i == np.floor(self.n_samples/4) or i == np.floor(self.n_samples/2) or i == np.floor(3*self.n_samples/4):
                print(' At interation {}'.format(i))

        # Basic summary statistics
        target_acceptance =  count / (self.n_samples)
        samples_reduced   = samples[self.burn_in:, :]
        mean = torch.mean(samples_reduced,dim=0, keepdim= True)
        print()
        print('****** EMPIRICAL MEAN/COV USING HMC ******')
        print('empirical mean : ', mean)
        print('Average acceptance rate is: ', target_acceptance)

        return samples,samples_reduced, mean
