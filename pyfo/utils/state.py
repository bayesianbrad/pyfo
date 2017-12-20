#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:12
Date created:  22/11/2017

License: MIT
'''
import torch
from torch.autograd import Variable
from collections import deque
from pyfo.utils.core import VariableCast
class State(object):
    """
    Stores the state of the object

    """

    def __init__(self, cls):
        """

        :param cls: this is the interface cls of the model.
        """

        self._state_init = cls.gen_prior_samples()
        self._gen_logpdf = cls.gen_pdf # returns logp
        self._cont_vars = cls.gen_cont_vars() #includes the piecewise variables for now.
        self._disc_vars = cls.gen_disc_vars()
        self._all_vars  = cls.gen_vars() # returns list of parameters, in same return order as self._state_init

    def intiate_state(self):
        """
        A dictionary of the state.
        :param
        :return: state type: intialized state
        """
        return self._state_init

    def _return_disc_list(self):
        if len(self._disc_vars) == 0:
            return None
        else:
            return self._disc_vars

    def _return_cont_list(self):
        if len(self._cont_vars) == 0:
            return None
        else:
            return self._cont_vars
    @staticmethod
    def detach_nodes(x):
        """
        Takes either the momentum or latents
        :param x:
        :return:
        """
        for key, value in x.items():
            x[key] = Variable(value.data, requires_grad=True)

    def _log_pdf(self, state, set_leafs= False):
        """
        The compiled pytorch function, log_pdf, should automatically
        return the pdf.
        :param
        :return: log_pdf

        Do I need to convert the variables within state, to requires grad = True
        here? Then they will be passed to gen_logpdf to create the differentiable logpdf
        . Answer: yes, because
        """
        if set_leafs:
            state = self._to_leaf(state)

        return self._gen_logpdf(state)
    def _log_pdf_update(self, state, step_size, log_prev, disc_params,j):
        """
        Implements the 'f_update' in the coordinate wise integrator, to calculate the
        difference in log probabiities.
        def f_update(x, dx, j, aux):

            logp_prev = aux

            x_new = x.clone()
            x_new.data[j] = x_new.data[j] + dx

            logp, _ , _  = f(x_new, False)
            logp_diff = logp - logp_prev
            aux_new = logp
            return logp_diff, aux_new

        :param state: dictionary of parameters
        :param step_size: torch.autograd.Variable
        :param log_prev: torch.autograd.Variable contains the value of the log_pdf before the change point
        :param disc_params: list of discrete params gen_sen keys
        :param j: permutated index, type int
        :return: log_diff , aux_new (the updated logp)?

        """
        # TO DO: Incorporate j parameter into this. See A3 Notes.
        update_parameter = disc_params[j]
        state[update_parameter] = state[update_parameter] + step_size
        current_logp = self._gen_logpdf(state)
        logp_diff = self.current_logpdf - log_prev
        return logp_diff, current_logp

    def _grad_logp(self, logp, param):
        """
        TO DO: Test, as this will only work for individual
        parameters. Will need to pass through an adapted version
        of state.
        Returns the gradient of the log pdf, with respect for
        each parameter
        :param state:
        :return: torch.autograd.Variable
        """

        gradient_of_param = torch.autograd.grad(outputs=logp, inputs=param, retain_graph=True)[0]
        return gradient_of_param

    def _to_leaf(self, state):
        """
        Ensures that all latent parameters are reset to leaf nodes, before
        calling
        :param state:
        :return:
        """
        for key in state:
            state[key] = VariableCast(state[key].data, grad=True)
        return state

    @staticmethod
    def _return_tensor(val1):
        """
        Takes a values, checks to see if it is a variable, if so returns tensor,
        else returns a tensor of the value
        :param val1:
        :return:
        """
        if isinstance(val1, Variable):
            return val1.data
        else:
            return val1