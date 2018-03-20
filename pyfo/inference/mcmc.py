#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  09:32
Date created:  20/03/2018

License: MIT
'''

from pyfo.inference.inference import Inference
from pyfo.pyfoppl.pyppl import compile_model
from pyfo.utils.core import transform_latent_support as tls

class MCMC(Inference):
    """

    """
    def __int__(self, debug_on):
        if debug_on:
            self.debug()

    def generate_model(self, model_code):
        '''
        Creates an inference algorithm.

        :param model_code
            type: str
            description:  This will interact with pyfoppl to generate a class, to compile the probabilistic program
            generate a graph describing that model. This class contains many methods for extracting latent parameters
            and generating the lod_Pdf.


        '''

        self.model = compile_model(model_code, base_class='base_model',
                                   imports='from pyfo.pyfoppl.pyppl.ppl_base_model import base_model')


    def generate_latent_vars(self):
        """

        :param args:
        :param kwargs:
        :return:
        """

        self._cont_latents = None if len(self.model.gen_cont_vars()) == 0 else self.model.gen_cont_vars
        self._disc_latents = None if len(self.model.gen_disc_vars()) == 0 else self.model.gen_disc_vars
        self._if_latents = None if len(self.model.gen_if_vars()) == 0 else self.model.gen_if_vars
        # each predicate has a cond boolean parameter, which represents the evaluated value of the predicate, whenever
        # the log_pdf is calculated.
        self._cond_bools = None if len(self.model.gen_if_vars()) == 0 else self.model.gen_cond_vars
        # all vertices of the graph are objects.
        self._vertices = self.model.get_vertices()
        # A list of all observables
        self.observables = dict([(vertex.name, vertex.observation) for vertex in self._vertices if vertex.is_observed])
        # a list of all vars
        self._all_vars = [vertex.name for vertex in self._vertices if vertex.is_sampled]

        # distribution type of each latent variable, used for the bijections and embeddings
        self._cont_dists = dict([(vertex.name, vertex.distribution_name) for vertex in self._vertices if
                                 (vertex.is_continuous and vertex.name in self._all_vars)])
        self._disc_dists = dict([(vertex.name, vertex.distribution_name) for vertex in self._vertices if
                                 (vertex.is_discrete and vertex.name in self._all_vars)])

        # discrete support sizes
        self._disc_support = dict(
            [(vertex.name, vertex.support_size) for vertex in self._vertices if vertex.is_discrete])

        # original parameter names. Parameter names are transformed in the compiler to ensure uniqueness
        self._names = dict(
            [(vertex.name, vertex.original_name) for vertex in self._vertices if vertex.name in self._all_vars])

    def initialize(self, n_iters=1000, n_chains=1,
                   auto_transform=True):
        '''
        Initialize inference algorithm. It initializes hyperparameters
        , the initial density and the values of each of the latents.

        :param n_iters
            type: int
            description: Number of iterations for algorithm when calling 'run()'.
            If called manually, it is the number of expected calls of 'update()';
            determines the tracking information during the print progress.
        :param n_chains
            type: int
            description: Number of chains for MCMC inference.

        :param debug:
            type: bool
            If true, prints out graphical model.
            TODO : Link with a tensorboard style framework
        '''

        self.n_iters = n_iters
        self.n_chains = n_chains
        self.auto_transform = auto_transform
        if self._cont_dists is not None:
            self.transforms = self.transform_check(self._cont_dists)
            # Returns {} of the support does not need to be changed.
        if self.transforms:
            self.state = self.model.gen_prior_samples()
            self._log_pdf = self.model.gen_pdf
        else:
            self.state  = self.model.gen_prior_samples_transformed()
            self._log_pdf = self.model.gen_pdf_transformed




    def debug(self):
        debug_prior = self.model.gen_prior_samples_code
        debug_pdf = self.model.gen_pdf_code
        print(50 * '=' + '\n'
                         'Now generating prior python code \n'
                         '{}'.format(debug_prior))
        print(50 * '=' + '\n'
                         'Now generating posterior python code \n'
                         '{}'.format(debug_pdf))
        print(50 * '=')
        print('\n Now generating graph code \n {}'.format(self.model))
        print(50 * '=')



    def transform_check(self, latent_vars):
        '''
        State whether or not the support will need transforming
        :param latent_vars
        :return: bool
        '''
        self.transforms = tls(self._cont_latents,self._cont_dists)


    def run_inference(self):
        '''

        :return:
        '''
