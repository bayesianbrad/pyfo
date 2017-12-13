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
from typing import Dict, List, bool
from pyfo.utils.core import VariableCast
class State(object):
    """
    Stores the state of the object

    """

    def __init__(self, interface):

        self._state_init = interface.gen_prior_samples()
        self._gen_logpdf = interface.gen_logpdf # returns logp
        self._cont_vars = interface.gen_cont_vars() #includes the piecewise variables for now.
        self._disc_vars = interface.gen_disc_vars()
        self._all_vars  = interface.gen_vars() # returns list of parameters, in same return order as self._state_init


    def intiate_state(self):
        """
        Creates a dictionary of the state. With each parameter transformed into a variable, if it is not already one.
        And ensures that the Variable is a leaf node
        :param
        :return: state type: Dict
        """
        state = dict.fromkeys(self._all_vars)
        values= deque(self._state_init)
        for var in state:
            state[var] = VariableCast(values.popleft().data, grad=True)

        return state

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

    def _log_pdf(self, state):
        """
        The compiled pytorch function, log_pdf, should automatically
        return the pdf.
        :param
        :return: log_pdf
        """
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