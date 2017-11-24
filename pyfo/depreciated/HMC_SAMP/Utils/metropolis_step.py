#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  09:03
Date created:  08/09/2017

License: MIT
'''

import torch
from torch.autograd import Variable
from Utils.kinetic import Kinetic

class Metropolis():

    def __init__(self, potential, integrator, M):
        self.potential  = potential
        self.integrator = integrator
        self.M          = M
        self.count      = 0

    def sample_momentum(self,values):
        assert(isinstance(values, Variable))
        return Variable(torch.randn(values.data.size()))

    def hamiltonian(self, logjoint, p):
        """Computes the Hamiltonian  given the current postion and momentum
        H = U(x) + K(p)
        U is the potential energy and is = -log_posterior(x)
        Parameters
        ----------
        logjoint    - Type:torch.autograd.Variable
                      Size: \mathbb{R}^{1 \times D}
        p           - Type: torch.Tensor.Variable
                      Size: \mathbb{R}^{1 \times D}.
                      Description: Auxiliary momentum
        log_potential :Function from state to position to 'energy'= -log_posterior

        Returns
        -------
        hamitonian : float
        """
        T = self.kinetic.gauss_ke(p, grad=False)
        return -logjoint + T
    def acceptance(self, values_init, logjoint_init, grad_init):
        '''Returns the new accepted state

        Parameters
        ----------

        Output
        ------
        returns accepted or rejected proposal
        '''

        # generate initial momentum

        #### FLAG
        p_init = self.sample_momentum(values_init)
        # generate kinetic energy object.
        self.kinetic = Kinetic(p_init, self.M)
        # calc hamiltonian  on initial state
        orig = self.hamiltonian(logjoint_init, p_init)

        # generate proposals
        values, p = self.integrator.leapfrog(p_init, values_init, grad_init)

        # calculate new hamiltonian given current
        logjoint_prop, _ = self.potential.eval(values, grad=False)

        current = self.hamiltonian(logjoint_prop, p)
        alpha = torch.min(torch.exp(orig - current))
        # calculate acceptance probability
        if isinstance(alpha, Variable):
            p_accept = torch.min(torch.ones(1, 1), alpha.data)
        else:
            p_accept = torch.min(torch.ones(1, 1), alpha)
        # print(p_accept)
        if p_accept[0][0] > torch.Tensor(1, 1).uniform_()[0][0]:  # [0][0] dirty code to get integersr
            # Updates count globally for target acceptance rate
            # print('Debug : Accept')
            # print('Debug : Count ', self.count)
            self.count = self.count + 1
            return values, self.count
        else:
            # print('Debug : reject')
            return values_init, self.count