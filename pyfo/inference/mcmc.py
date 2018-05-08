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

from torch.multiprocessing import cpu_count
import torch.multiprocessing as mp
import numpy as np
import sys
import os
# import multiprocessing as pmp
# import pathos.pools as pp
# import dill as pickle
import pickle
import pathlib
import time
import xarray


class MCMC(Inference):
    """
    General purpose MCMC inference base class
    """

    def __init__(self, model_code=None, generate_graph=False, debug_on=False, dir_name=sys.path[0]):
        self.generate_graph = generate_graph
        self.debug_on = debug_on
        self.model_code = model_code
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

        print(5 * '-' + ' Intializing the inference ' + 5 * '-')
        self.auto_transform = True
        # if isinstance(self.kernel, HMC):
        #     self.auto_transform = True
        # else:
        #     self.auto_transform = False
        if self._cont_latents is not None:
            self.transforms = tls(self._cont_latents, self._cont_dists)

        # else: # The following is redundant for now.
        #         #     self.state  = self.model.gen_prior_samples_transformed()
        #         #     self._gen_log_pdf = self.model.gen_pdf_transformed
        self.state = self.model.gen_prior_samples()

        if self.generate_graph:
            print(50 * '-')
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


    def run_inference(self, kernel=None, nsamples=1000, burnin=100, chains=1, warmup=100, step_size=None,
                      num_steps=None, adapt_step_size=True, trajectory_length=None):
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
        :param dirname :type: str descrip: Path to a directory, where data can be saved. Default is the directory in
        which the code is run.

        :return: samples, :type pandas.dataframe

        '''
        self.kernel = kernel if not None else warnings.warn('You must enter a valid kernel')
        AVAILABLE_CPUS = cpu_count()

        #
        def run_sampler(state, nsamples, burnin, chain, UNIQUE_ID, snamepick, snamepd):
            samples_dict = []

            pathlib.Path(dir_n).mkdir(parents=True, exist_ok=True)
            if not isinstance(state, dict):
                sample = state.get()
            else:
                sample = state
            samples_dict.append(sample)
            snamepick = snamepick + str(UNIQUE_ID) + '_chain_' + str(chain) + '.pickle'
            snamepd = snamepd + str(UNIQUE_ID) + '_chain_' + str(chain) + '.csv'
            with open(snamepick, 'wb') as fout:
                for ii in tqdm(range(nsamples + burnin - 1)):
                    sample = self._instance_of_kernel.sample(sample)
                    samples_dict.append(sample)
                    pickle.dump(samples_dict, fout)

                samples = pd.DataFrame.from_dict(samples_dict, orient='columns').rename(columns=self._instance_of_kernel._names, inplace=True)
                print(50 * '=', '\n Saving xarray dataframe to : {0} '.format(snamepd))
        #
                xarray.DataArray,from_dict(snamepd, index=False, header=True)
                print(50* '=')

        # convert_to_numpy or just write own function for processing a dataframe full of
        # tensors? May try the later approach
        # Note : The pd.dataframes contain torch.tensors
            if not isinstance(state, dict):
                sys.stdout.write('The type of q is : {} '.format(type(state)))
                state.put(samples_dict)
            else:
                sys.stdout.write('The type of q is : {} '.format(type(state)))
                return samples_dict

        # Generate instance of transition kernel

        self._instance_of_kernel = self.kernel(model_code=self.model_code, step_size=step_size, num_steps=num_steps,
                                               adapt_step_size=adapt_step_size, trajectory_length=trajectory_length, \
                                               generate_graph=self.generate_graph, debug_on=self.debug_on)
        self._instance_of_kernel.setup(state=self._instance_of_kernel.state, warmup=warmup)

        # save  samples paths
        dir_n = os.path.join(self._dir_name, 'results')
        UNIQUE_ID = np.random.randint(0, 1000)
        snamepick = os.path.join(dir_n, 'samples_')
        snamepd = os.path.join(dir_n, 'all_samples_' + str(UNIQUE_ID))
        print('Saving individual samples in:  {0} \nwith unique ID: {1}'.format(dir_n, UNIQUE_ID))


        print(50 * '-')
        if chains > 1:
            print(5 * '-' + ' Generating samples for {0} chains {1}'.format(chains, 5*'-'))
            print(50 * '-')
            q = mp.Queue()
            processes = []
            for rank in range(AVAILABLE_CPUS):
                q.put(self._instance_of_kernel.state)
                p = mp.Process(target=run_sampler, args=(q, nsamples, burnin, rank, UNIQUE_ID, snamepick, snamepd))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            samples_ = [q.get() for _ in range(AVAILABLE_CPUS)]
        else:
            print(5 * '-' + ' Generating samples for {0} chain {1}'.format(chains, 5*'-'))
            print(50 * '-')
            samples_ = run_sampler(self._instance_of_kernel.state, nsamples, burnin, chains, UNIQUE_ID, snamepick, snamepd) # runs a single chain

        return samples_


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)