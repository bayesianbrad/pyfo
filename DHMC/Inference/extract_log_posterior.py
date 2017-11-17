#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:21
Date created:  16/11/2017

License: MIT
'''

'''
A script to extract the output from the foppl compiler

Inputs:

Assume that we already know the output to the FOPPL, which we specify via PATH
    PATH = ../DHMC/models/<model_name>.py
    

Outputs:

log_pdf of the model
gradient of the model

'''

import torch
from torch.autograd import Variable
from DHMC.utils.core import VariableCast
import importlib

def my_import(name):
    '''
    Helper function for extracting the whole module and not just the package.
    See answer by clint miller for details:
    https://stackoverflow.com/questions/951124/dynamic-loading-of-python-modules

    :param name
    :type string
    :return module
    '''
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
class state(object):
    '''
    Contains the state of the model
    '''
    def __init__(self, PATH_TO_MODEL):
        self._model  = my_import(PATH_TO_MODEL)
        self._priors = self._model.gen_prior_samples() # returns list of prior samples.
        self._posterior = self._model.gen_pdf # to this we need to pass _priors, but ensure the vairables have requires
        # grad = True. and can pass a compute grad flag too.
        self._allVars = self._model.gen_ordered_vars()
        self._contVars = self._model.gen_cont_vars()
        self._discVars = self._model.gen_disc_vars()


    def enableGrad_priors(self):
        """ Takes the Xs values and then """
        return 0
