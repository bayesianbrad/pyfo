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
    Stores the state of the object in

    """

    def __init__(self, gen_prior_samples, gen_logpdf, gen_cont_vars, gen_disc_vars, gen_all_vars):

        self._state_init = gen_prior_samples()
        self._gen_logpdf = gen_logpdf # returns logp
        self._cont_vars = gen_cont_vars() #includes the piecewise variables for now.
        self._disc_vars = gen_disc_vars()
        self._all_vars  = gen_all_vars() # returns list of parameters, in same return order as self._state_init


    def _intiate_state(self):
        """
        Creates a dictionary of the state. With each parameter transformed into a variable, if it is not already one.
        :param
        :return: state type: Dict
        """
        state = dict.fromkeys(self._all_vars)
        values= deque(self._state_init)
        for var in state:
            state[var] = VariableCast(values.popleft())

        return state

    @staticmethod
    def retain_grads(x):
        """
        Takes either the momentum or latents, checks to see if they are a leaf node,
        if so, adds requires_grad = True
        :param x: Dict of parameter names and values, values typically will be variables.
        :return: a
        """
        for value in x.values():
            # XXX: can be removed with PyTorch 0.3
            if value.is_leaf and not value.requires_grad:
                value.requires_grad = True
            value.retain_grad()
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
        This needs convert the state map, then run self._gen_pdf

        :param
        :return: log_pdf
        """

        self.current_logpdf = self._gen_logpdf(state)
        return self._gen_logpdf()
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
        logp_diff = self.current_logpdf - log_prev
        return logp_diff, self.current_logpdf

    def _grad_potential(self, state):
        log_joint_prob = self._log_pdf(state)
        log_joint_prob.backward()
        grad_potential = {}
        for name, value in state.items():
            grad_potential[name] = -value.grad.clone().detach()
            grad_potential[name].volatile = False
        return grad_potential


    def _state_grad(self):
        """
        Calculates the gradient
        :param compute_grad: type: Variable
        :return: gradients
        """
        # grad = torch.autograd.grad(logp, var_cont)[0]  # need to modify format

