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
from  ..pyppl import compile_model
from  .inference import Inference
from ..utils.core import transform_latent_support as tls
from ..utils.core import _to_leaf, convert_dict_vars_to_numpy, create_network_graph, display_graph
from tqdm import tqdm
import torch.distributions as dists_
from torch.multiprocessing import cpu_count
import torch.multiprocessing as mp
import numpy as np
import sys
import os
from ..utils.eval_stats import data_summary as ds
import pickle
import pathlib
import time
# import xarray
import warnings


class MCMC(Inference):
    """
    General purpose MCMC inference base class
    """

    def __init__(self, model_code=None, generate_graph=False, debug_on=False, model_name=None, dir_name=sys.path[0]):
        self.generate_graph = generate_graph
        self.debug_on = debug_on
        self.model_code = model_code
        self._dir_name = dir_name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = 'unique'
        # warnings.warn('be careful about using input.sum() as model code should have .sum() in gen_log_pdf')
        super(MCMC, self).__init__()

    def generate_model(self, model_code):
        '''
        Creates an inference algorithm.

        :param str model_code: This will interact with pyfoppl to generate a class, to compile the probabilistic program
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
                          'method. See ..pyfoppl.pyppl.ppl_base_model for more info on method names')
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
        # print('Debebug statement in MCMC.generate_latents() \n Printing if vars : {} '.format(self._if_latents))
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
        self._ancestors = {}
        for v in self._vertices:
            if v.is_sampled:
                for a in v.ancestors:
                    self._ancestors[a.name] =  v.name




        # original parameter names. Parameter names are transformed in the compiler to ensure uniqueness
        self._names = dict(
            [(vertex.name, vertex.original_name) for vertex in self._vertices if vertex.name in self._all_vars])
        # distribution arguments and names
        self._dist_params = {}
        for vertex in self._vertices:
            if vertex.is_sampled:
                self._dist_params[vertex.name] = {vertex.distribution_name: vertex.distribution_arguments}

        # TODO fix this function self._disc_support = dict()
        # key: values of parameter name and string of distribution object.
        self._dist_obj = {}
        if self._disc_latents:
            for param in self._disc_latents:
                for dist in self._dist_params[param]:
                    i = 0
                    self._dist_obj[param] = ''
                    for key in [self._dist_params[param][dist]]:
                        self._dist_obj[param] = self._dist_obj[param] + key + '=' + self._dist_params[param][dist][key]
                        i += 1
                        if len(self._dist_params[param][dist].keys()) > 1 and i == 1:
                            self._dist_obj[param] = self._dist_obj[param] + ','
                        if len(self._dist_params[param][dist].keys()) > 2 and i == 2:
                            self._dist_obj[param] = self._dist_obj[param] + ','
                    self._dist_obj[param] = dist + '( {0} )'.format(self._dist_obj[param])
            # discrete support sizes using the self.__dist_obj get the support of each
            # distribution by reconstructing it. Might be able to include this in the outer for loop.
            self._discrete_support = {}

    def initialize(self):
        '''
        Initialize inference algorithm. It initializes hyperparameters, the initial density and the values of each of the latents.

        :param int n_iters:  Number of iterations for algorithm when calling `run()`. If called manually, it is the number of expected calls of `update()` determines the tracking information during the print progress.
        :param int n_chains: Number of chains for MCMC inference.
        :param bool debug: If true, prints out graphical model.

        '''

        print(5 * '-' + ' Intializing the inference ' + 5 * '-')
        if self.debug_on:
            print(50 * '-')
            print('Debug function printing .....')
            print(50 * '-' + '\n'
                             'Now generating compiled model code output \n'
                             '{}'.format(self.model.code))
            print(50 * '-')
            print(50 * '-' + '\n'
                             'Now generating vertices \n'
                             '{}'.format(self._vertices))
            print(50 * '-')
            print('End of debug function printing .....')
            print(50 * '-')

        if self.generate_graph:
            print(50 * '-')
            print('{0} Generating an a graph G(V,E) of your model {0}'.format('-'))
            create_network_graph(vertices=self._vertices)
            display_graph(vertices=self._vertices)
            print(50 * '-')


        self.auto_transform = True

    
        if self._cont_latents is not None:
            self.transforms = tls(self._cont_latents, self._cont_dists)

    def warmup(self):
        return 0


    def run_inference(self, kernel=None, nsamples=1000, burnin=100, chains=1, warmup=100, step_size=None,
                      num_steps=None, adapt_step_size=True, trajectory_length=None):
        """
        The run inference method should be run externally once the class has been created.
        I.e assume that they have not written there own model.
        It then returns the samples generated from inference. Alternatively you can set a global directory for
        saving plots and samples generated.

        :param int nsamples: Specifies how many samples you would like to generate.
        :param int burnin: Specifies how many samples you would like to remove.
        :param int chains: Specifies the number of chains.
        :param float step_size: Specifies the sized step in inference.
        :param int num_steps: The trajectory length of the inference algorithm
        :param bool adapt_step_size: Specifies whether you would like to use auto-tune features
        :param int trajectory_length: Specfies the legnth of the leapfrog steps
        :param str dirname: Path to a directory, where data can be saved. Default is the directory in which the code is run.

        Example:

            .. code-block:: python

                    >>> hmc = MCMC('HMC')
                    >>> samples = hmc.run_inference(nsamples=1000,\
                    >>> burnin=100,\
                    >>> chains=1,\
                    >>> warmup= 100,\
                    >>> step_size=None,\
                    >>> num_steps=None,\
                    >>> adapt_step_size=False,\
                    >>> trajectory_length = None,\
                    >>> dirname = None)


        """

        self.kernel = kernel if not None else warnings.warn('You must enter a valid kernel')
        self.burnin = burnin
        AVAILABLE_CPUS = cpu_count()
        
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
            # with open(snamepick, 'wb') as fout:
            for _ in tqdm(range(nsamples + burnin - 1)):
                # print('Debug statement in run_sampler \n Printing initial sample :\n {}'.format(sample))
                sample = self._instance_of_kernel.sample(sample)
                # print('Debug statement in run_sampler \n Printing proposed sample :\n {}'.format(sample))
                samples_dict.append(sample)
                    # pickle.dump(samples_dict, fout)

                # samples = pd.DataFrame.from_dict(samples_dict, orient='columns').rename(columns=self._instance_of_kernel._names, inplace=True)

            if not isinstance(state, dict):
                # sys.stdout.write('The type of q is : {} '.format(type(state)))
                state.put(samples_dict)
            else:
                # sys.stdout.write('The type of q is : {} '.format(type(state)))
                return samples_dict

        # Generate instance of transition kernel

        self._instance_of_kernel = self.kernel(model_code=self.model_code, step_size=step_size, num_steps=num_steps,
                                               adapt_step_size=adapt_step_size, trajectory_length=trajectory_length, \
                                               generate_graph=self.generate_graph, debug_on=self.debug_on)
        self.state = []
        for i in range(chains):
            self.state.append(self._instance_of_kernel.model.gen_prior_samples())
        # save  samples paths
        dir_n = os.path.join(self._dir_name, 'results')
        UNIQUE_ID = np.random.randint(0, 1000)
        snamepick = os.path.join(dir_n, self.model_name + '_samples_')
        snamepd = os.path.join(dir_n, self.model_name + '_samples_' + str(UNIQUE_ID))
        print('Saving {0} model samples in:  {1} \nwith unique ID: {2}'.format(self.model_name, dir_n, UNIQUE_ID))

        print(50 * '-')
        if chains > 1:
            print(5 * '-' + ' Generating samples for {0} chains {1}'.format(chains, 5 * '-'))
            print(50 * '-')
            q = mp.Queue()
            processes = []
            for rank in range(chains):
                q.put(self.state[rank])
                p = mp.Process(target=run_sampler, args=(q, nsamples, burnin, rank, UNIQUE_ID, snamepick, snamepd))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            samples_ = [q.get() for _ in range(chains)]
        else:
            print(5 * '-' + ' Generating samples for {0} chain {1}'.format(chains, 5 * '-'))
            print(50 * '-')
            samples_ = [run_sampler(self.state[chains-1], nsamples, burnin, chains, UNIQUE_ID, snamepick,
                                   snamepd)]  # runs a single chain
        self.samples = samples_
        self._names = self._instance_of_kernel._names
        return samples_

    def return_statistics(self):
        ''' Returns dictionary of samples with burnin, a dictionary of means and a dictionary of variancea summary stats '''

        samples, means, variances, std =  ds(samples=self.samples, burnin=self.burnin, true_names=self._names)
        print(70*'-')
        print('{0} Printing summary statistics {0}'.format(10*'-'))
        print(70*'-')
        import texttable as tt
        tab = tt.Texttable()
        headings = ['Chain', 'Parameter', 'Mean', 'Variance', 'Std']
        tab.header(headings)
        chains = []
        params = []
        means_ = []
        vars_ = []
        std_ =[]
        for chain in means:
            chains.append(chain)
            for key in means[chain]:
                params.append(key)
                means_.append(means[chain][key])
                vars_.append(variances[chain][key])
                std_.append(std[chain][key])
        for row in zip(chains, params, means_, vars_, std_):
            tab.add_row(row)

        table = tab.draw()
        print(table)

        return samples, means, variances, std


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