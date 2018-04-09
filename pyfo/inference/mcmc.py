#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  09:32
Date created:  20/03/2018

License: MIT
'''
import torch
import math
import inspect
import warnings
from pyfo.inference.inference import Inference
from pyfo.pyfoppl.pyppl import compile_model
from pyfo.utils.core import transform_latent_support as tls
from pyfo.utils.core import _to_leaf

class MCMC(Inference):
    """

    """
    def __int__(self, kernel, model_code, debug_on=False):
        if debug_on:
            self.debug()
        self.kernel = kernel
        if inspect.isclass(model_code):
            # check to see if user has overrideded base class
            self.model = model_code
            warnings.warn('You have overridden the base class, please ensure \n '
                          'that your model complies with the standard compiler output.\n '
                          'In regard to the unique parameter names. If you need to determine\n '
                          'what these are. compile_model(model_code).<methods> and choose the appropriate \n'
                          'method. See pyfo.pyfoppl.pyppl.ppl_base_model for more info on method names')
        else:
            self.model_code = model_code
            self.generate_model()

        self.generate_latent_vars()
        self.initialize()
        self.run_inference(kernel=kernel,nsamples=nsamples,burnin=burnin,chains=chains)
    def generate_model(self):
        '''
        Creates an inference algorithm.

        :param model_code
            type: str
            description:  This will interact with pyfoppl to generate a class, to compile the probabilistic program
            generate a graph describing that model. This class contains many methods for extracting latent parameters
            and generating the lod_Pdf.


        '''

        self.model = compile_model(self.model_code, base_class='base_model',
                                   imports='from pyfo.pyfoppl.pyppl.ppl_base_model import base_model')


    def generate_latent_vars(self):
        """
        You receive all the information regarding the whole model,
        including model elements that are not common among all inference
        algorithms. This ensures flexibility, as all inference procedures
        inherit from this class. It then allows them to manipulate
        quantities as required. Doing this additional set-up has very
        minimal computational cost.


        :return: Attributes to the MCMC class all model parameters and
                 types
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

        # distribution arguments and names
        self._dist_params = {}
        for vertex in self._vertices:
            if vertex.is_sampled:
                self._dist_params[vertex.name] = {vertex.distribution_name: vertex.distribution_arguments}


    def initialize(self, auto_transform=True):
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

        self.auto_transform = auto_transform
        if self._cont_dists is not None:
            self.transforms = self.transform_check(self._cont_dists)
            # Returns {} ff the support does not need to be changed.
        if self._disc_dists is not None:
            self._disc_support
        if self.transforms:
            self.state = self.model.gen_prior_samples()
            self._gen_log_pdf = self.model.gen_pdf
        else:
            self.state  = self.model.gen_prior_samples_transformed()
            self._gen_log_pdf = self.model.gen_pdf_transformed




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


    def run_inference(self, nsamples=1000,burnin=100,chains=1):
        '''

        :return:
        '''

    def __grad_logp(self, logp, param):
        """
        Returns the gradient of the log pdf, with respect for
        each parameter. Note the double underscore, this is to ensure that if
        this method is overwritten, then no problems occur when overidded.
        :param state:
        :return: torch.autograd.Variable
        """
        gradient_of_param = torch.autograd.grad(outputs=logp, inputs=param, retain_graph=True)[0]
        return gradient_of_param


    def __generate_log_pdf(self, state, set_leafs=False):
        """
        The compiled pytorch function, log_pdf, should automatically
        return the pdf.
        :param keys type: list of discrete embedded discrete parameters
        :return: log_pdf

        Maybe overidden in other methods, that require dynamic pdfs.
        For example
        if you have a model called my mymodel, you could write the following:
        Model = compile_model(mymodel) # returns class
        class MyNewModel(Model):

            def gen_log_pdf(self, state):
                for vertex in self.vertices:
                    pass
        return "Whatever you fancy"

        # This overrides the base method.
        # Then all you have to do is pass
        # My model into kernel of choice, i.e
        hmc_kern = hmc(MyNewModel, ....)
        hmc_kern.sample()


        """
        if set_leafs:
            state = _to_leaf(state)

        return self._gen_logpdf(state)

