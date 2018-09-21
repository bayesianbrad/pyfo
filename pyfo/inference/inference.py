#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:01
Date created:  06/03/2018

License: MIT
'''
import torch
from abc import abstractmethod

class Inference(object):

    '''
    Abstract base class for all inference methods. All inference methods will inherit from this class and will thus share
    all common methods and attributes.


    '''
    @abstractmethod
    def generate_model(self, *args, **kwargs):
        """
        Creates the model code. This will interact with `pysppl` to generate a class, to compile the
        probabilistic program generate a graph describing that model. This class, depending on what
        compiler you use, should contains methods for extracting latent parameters and generating the `lod_Pdf`.
        See `MCMC.py` for more details.

        :param str model_code:
        """
        raise NotImplementedError


    @abstractmethod
    def generate_latent_vars(self, *args, **kwargs):
        '''
        See `MCMC.py` for ideas.

        :param cls model: contains methods for extracting latent variables from the DAG directed acyclic graph.

        .. note::
             Although the user can implement their own graph and pass that to this inference engine. Provided there
             exists individual lists of the continuous, discrete and conditional variables as strings.

        Example::

            >>>cont_vars = ['x1', 'x2'] , disc_vars = ['x3'], cond_vars = ['x4']

        For each `cond_var` there exists a predicate which must also be handled correctly. You will need a dictionary
        of pairs `{'condition": boolean}` of the conditioning statement and the boolean value associated with that statement

        Example::

             >>>{'cond101': True} comes from
             >>>x4 ~ sample(Normal(0,1))
             >>>if  x4 > 0 :
             >>>    observe(N(1,0,1))
             >>>else:
             >>>    observe(N(-1,0,2))
             >>>
             >>>'cond101' == x4 - 0
             >>> Assume x4 = 1.73121, therefore {'cond101': True}



        .. note::
            You will also need the len_of_support of the discrete parameters,
            as pairs `{'str_of_discrete_latent' : len_of_support}`

        Example::

            >>>if x3 ~ sample(cat[0.1, 0.5, 0.4]) support = 3 ==> x3 \in [0,2]

        .. note::
            Depending on how you implement the unique names for generating your model and using those parameters in the
            inference, you will need a dict of pairs `{'original_name':'inference_name'}`.

        '''

    @abstractmethod
    def initialize(self, *args, **kwargs):
        '''

        :return:
        '''

        raise NotImplementedError

    @abstractmethod
    def run_inference(self, *args, **kwargs):

        raise NotImplementedError
