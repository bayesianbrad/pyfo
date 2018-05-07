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
import pandas as pd
from pyfo.inference.inference import Inference
from pyfo.pyfoppl.pyppl import compile_model
from pyfo.utils.core import transform_latent_support as tls
from pyfo.utils.core import _to_leaf, convert_dict_vars_to_numpy, create_network_graph, display_graph
from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# from multiprocessing.pool import ApplyResult
import numpy as np
import sys
import os
import pathos.multiprocessing as pmp
import pathos.pools as pp
# import dill as pickle
import pickle
import pathlib
import time

class MCMC(Inference):
    """
    General purpose MCMC inference base class
    """
    def __init__(self, model_code=None, generate_graph=False, debug_on=False, save_data=False, dir_name=sys.path[0]):
        self.generate_graph=  generate_graph
        self.debug_on = debug_on
        self.model_code = model_code
        self._save_data =save_data
        self._dir_name = dir_name
        super(MCMC, self).__init__()



    def generate_model(self, model_code):
        '''
        Creates an inference algorithm.

        :param model_code
            type: str
            description:  This will interact with pyfoppl to generate a class, to compile the probabilistic program
            generate a graph describing that model. This class contains many methods for extracting latent parameters
            and generating the lod_Pdf.


        '''

        return compile_model(model_code)


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
        print(50 * '-')
        print(5 * '-' + ' Compiling model ' + '-' * 5)
        print(50 * '-')
        if inspect.isclass(self.model_code):
            start = time.time()
            # check to see if user has overrideded base class
            self.model = self.model_code
            warnings.warn('You have overridden the base class, please ensure \n '
                          'that your model complies with the standard compiler output.\n '
                          'In regard to the unique parameter names. If you need to determine\n '
                          'what these are. compile_model(model_code).<methods> and choose the appropriate \n'
                          'method. See pyfo.pyfoppl.pyppl.ppl_base_model for more info on method names')
            end = time.time()
        else:
            start = time.time()
            self.model = self.generate_model(model_code=self.model_code)
            end = time.time()
        print('\n Model complied in {0} seconds \n '.format(end - start))
        print(50 * '-')

        self._cont_latents = None if len(self.model.gen_cont_vars()) == 0 else self.model.gen_cont_vars()
        self._disc_latents = None if len(self.model.gen_disc_vars()) == 0 else self.model.gen_disc_vars()
        self._if_latents = None if len(self.model.gen_if_vars()) == 0 else self.model.gen_if_vars()
        # each predicate has a cond boolean parameter, which represents the evaluated value of the predicate, whenever
        # the log_pdf is calculated.
        self._cond_bools = None if len(self.model.gen_if_vars()) == 0 else self.model.gen_cond_vars()
        # all vertices of the graph are objects.
        self._vertices = self.model.get_vertices()
        # A list of all observables
        self.observables = dict([(vertex.name, vertex.observation) for vertex in self._vertices if vertex.is_observed])
        # a list of all latent vars
        self._all_vars = [vertex.name for vertex in self._vertices if vertex.is_sampled]
        self._number_of_latents = len(self._all_vars)
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

    def initialize(self):
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

        '''

        print( 5 * '-' + ' Intializing the inference ' + 5 * '-')
        self.auto_transform = True
        # if isinstance(self.kernel, HMC):
        #     self.auto_transform = True
        # else:
        #     self.auto_transform = False
        if self._cont_latents is not None:
            self.transforms = tls(self._cont_latents,self._cont_dists)

        # else: # The following is redundant for now.
        #         #     self.state  = self.model.gen_prior_samples_transformed()
        #         #     self._gen_log_pdf = self.model.gen_pdf_transformed
        self.state = self.model.gen_prior_samples()

        if self.generate_graph:
            print(50*'-')
            print('{0} Generating an a graph G(V,E) of your model {0}'.format('-'))
            create_network_graph(vertices=self._vertices)
            display_graph(vertices=self._vertices)
            print(50 * '-')

        if self.debug_on:
            print(50 * '=' + '\n'
                             'Now generating compiled model code output \n'
                             '{}'.format(self.model.code))
            print(50 * '=')


    def warmup(self):
        return 0

    def run_inference(self, kernel=None, nsamples=1000, burnin=100, chains=1, warmup=100, step_size=None,  num_steps=None, adapt_step_size=True, trajectory_length=None):
            '''
            The run inference method should be run externally once the class has been created.
            I.e assume that they have not written there own model.

            hmc = MCMC('HMC')
            # all of the following kwargs are optional and kernel dependent.
            samples = hmc.run_inference(nsamples=1000,
                                        burnin=100,
                                        chains=1,
                                        warmup= 100,
                                        step_size=None,
                                        num_steps=None,
                                        adapt_step_size=False,
                                        trajectory_length = None,
                                        save_data= False
                                        dirname = None)

            It then returns the samples generated from inference. Alternatively you can set a global directory for
            saving plots and samples generated.

            :param nsamples type: int descript: Specifies how many samples you would like to generate.
            :param burnin: type: int descript: Specifies how many samples you would like to remove.
            :param chains :type: int descript: Specifies the number of chains.
            :param step_size: :type: float descript: Specifies the sized step in inference.
            :param num_steps :type: int descript: The trajectory length of the inference algorithm
            :param adapt_step_size :type: bool descript: Specifies whether you would like to use auto-tune features
            :param trajectory_length :type int descript: Specfies the legnth of the leapfrog steps
            :param save_data :type bool descrip: Specifies whether to save data and return data, or just return.
            :param dirname :type: str descrip: Path to a directory, where data can be saved. Default is the directory in
            which the code is run.

            :return: samples, :type pandas.dataframe

            '''
            self.kernel = kernel if not None else warnings.warn('You must enter a valid kernel')
            AVAILABLE_CPUS = pmp.cpu_count()

            def run_sampler(state, nsamples, burnin, chain, save_data=self._save_data, dir_name=self._dir_name):
                samples_dict = []

                if save_data:
                    dir_n = os.path.join(dir_name,'results')
                    pathlib.Path(dir_n).mkdir(parents=True, exist_ok=True)
                    UNIQUE_ID = np.random.randint(0,1000)
                    snamepick = os.path.join(dir_n,'samples_' + str(UNIQUE_ID) + '_chain_' + str(chain)+'.pickle')
                    snamepd =  os.path.join(dir_n,'all_samples_' + str(UNIQUE_ID) + '_chain_' + str(chain) + '.csv')
                    # Generates prior sample - the initliaziation of the state
                    # print('Debug statement in run_sampler() . Printing state : {0}'.format(state))
                    sample= state
                    samples_dict.append(sample)
                    print('Saving individual samples in:  {0} \n with unique ID: {1}'.format(dir_n, UNIQUE_ID))
                    # try:
                    with open(snamepick, 'wb') as fout:
                        for ii in tqdm(range(nsamples+burnin - 1)):
                            sample = self._instance_of_kernel.sample(sample)
                            # print('Debug statement in run_sampler() : \n Printing samples : {}'.format(sample))
                            samples_dict.append(sample)
                            pickle.dump(samples_dict, fout)

                    samples = pd.DataFrame.from_dict(samples_dict, orient='columns').rename(
                        columns=self._instance_of_kernel._names, inplace=True)
                    print('Debug statement in run_sampler() : \n Printing true names: {}'.format(self._instance_of_kernel._names))
                    print(50 * '=', '\n Saving pandas dataframe to : {0} '.format(snamepd))

                    samples.to_csv(snamepd, index=False, header=True)

                # except Exception:
                #TODO : convert the pickle samples to dataframe.
                #TODO: if the fucntion  is stopped for whatever reason, trigger a separate function that takes
                # the saved msg, upacks it and returns the dataframe with correct
                # latent variable names.
                else:
                    # print('Debug statement in run_sampler() . Printing state : {0}'.format(state))
                    sample = state
                    samples_dict.append(sample)
                    for ii in tqdm(range(nsamples+burnin - 1)):
                        sample = self._instance_of_kernel.sample(sample)
                        samples_dict.append(sample)
                    samples = pd.DataFrame.from_dict(samples_dict, orient='columns', dtype=float).rename(
                        columns=self._instance_of_kernel._names, inplace=True)

                # convert_to_numpy or just write own function for processing a dataframe full of
                # tensors? May try the later approach
                # Note : The pd.dataframes contain torch.tensors


                # samples.rename(columns=self._names, inplace=True) the above may not work.
                return samples


            self._instance_of_kernel = self.kernel(model_code=self.model_code, step_size=step_size,  num_steps=num_steps, adapt_step_size=adapt_step_size, trajectory_length=trajectory_length,\
                                                   generate_graph = self.generate_graph, debug_on= self.debug_on)
            self._instance_of_kernel.setup(state=self._instance_of_kernel.state, warmup=warmup)
            print(50*'-')
            print(5*'-' + ' Generating samples ' + 5*'-')
            print(50 * '-')
            if chains > 1:
                pool = pmp.Pool(processes=AVAILABLE_CPUS)
                samples = [pool.apply_async(run_sampler, (self._instance_of_kernel.state, nsamples, burnin, chain)) for chain in range(chains)]  #runs multiple chains in parallel
                samples = [chain_.get() for chain_ in samples]
            else:
                samples = run_sampler(state=self._instance_of_kernel.state, nsamples=nsamples, burnin=burnin, chain=chains) # runs a single chain


            return samples

    def _grad_logp(self, logp, param):
        """
        Returns the gradient of the log pdf, with respect for
        each parameter. Note the double underscore, this is to ensure that if
        this method is overwritten, then no problems occur when overidded.
        :param state:
        :return: torch.autograd.Variable
        """

        gradient_of_param = torch.autograd.grad(outputs=logp, inputs=param, retain_graph=True)[0]
        return gradient_of_param


    def _generate_log_pdf(self, state, set_leafs=False):
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
        kernel = MCMC(MyNewModel,kernel=HMC)
        kernel.run_inference()


        """
        if set_leafs:
            # only sets the gradients of the latent variables.
            _state = _to_leaf(state=state, latent_vars=self._all_vars)
        else:
            _state = state
        return self.model.gen_log_pdf(_state)

