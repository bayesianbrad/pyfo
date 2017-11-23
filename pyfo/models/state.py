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
from typing import Dict, List, Bool
from pyfo.utils.core import VariableCast

class State(object):
    """
    Stores the state of the object in

    """

    def __init__(self, gen_prior_samples, gen_pdf, gen_cont_vars, gen_disc_vars):

        self.state = gen_prior_samples
        self._gen_pdf = gen_pdf # returns logp
        # TO DO: Tell Yuan not to only return logp
        self._cont_vars = gen_cont_vars #includes the piecewise variables for now.
        self._disc_vars = gen_disc_vars


    def _intiate_state(self):
        """
        :param
        :return:
        """
        state = self._gen_pdf
        return state

    def _to_Variable(self):
        """
        converts the vars of the state to floats, by extracting the data.
        :param state
        :return:
        """
    def _log_pdf(self, state):
        """
        This needs convert the state map

        :param
        :return: log_pdf
        """

    def _state_grad(self):
        """
        Calculates the gradient
        :param compute_grad: type: Variable
        :return: gradients
        """
        # grad = torch.autograd.grad(logp, var_cont)[0]  # need to modify format

