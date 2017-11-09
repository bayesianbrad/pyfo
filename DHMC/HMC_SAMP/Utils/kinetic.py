#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  18:33
Date created:  01/09/2017

License: MIT
'''
from core import VariableCast
import torch
from torch.autograd import Variable
class Kinetic():
    ''' A basic class that implements kinetic energies and computes gradients
    Methods
    -------
    gauss_ke          : Returns KE gauss
    laplace_ke        : Returns KE laplace

    Attributes
    ----------
    p    - Type       : torch.Tensor, torch.autograd.Variable,nparray
           Size       : [1, ... , N]
           Description: Vector of current momentum

    M    - Type       : torch.Tensor, torch.autograd.Variable, nparray
           Size       : \mathbb{R}^{N \times N}
           Description: The mass matrix, defaults to identity.

    '''
    def __init__(self, p, M = None):

        if M is not None:
            if isinstance(M, Variable):
                self.M  = VariableCast(torch.inverse(M.data))
            else:
                self.M  = VariableCast(torch.inverse(M))
        else:
            self.M  = VariableCast(torch.eye(p.size()[0])) # inverse of identity is identity


    def gauss_ke(self,p, grad = False):
        '''' (p dot p) / 2 and Mass matrix M = \mathbb{I}_{dim,dim}'''
        self.p = VariableCast(p)
        P = Variable(self.p.data, requires_grad=True)
        K = 0.5 * P.t().mm(self.M).mm(P)

        if grad:
            return self.ke_gradients(P, K)
        else:
            return K
    def laplace_ke(self, p, grad = False):
        self.p = VariableCast(p)
        P = Variable(self.p.data, requires_grad=True)
        K = torch.sign(P).mm(self.M)
        if grad:
            return self.ke_gradients(P, K)
        else:
            return K
    def ke_gradients(self, P, K):
        return torch.autograd.grad([K], [P])[0]