#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:01
Date created:  06/03/2018

License: MIT
'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:13
Date created:  05/03/2018

License: MIT

import tensorflow as tf
from edward.models.random_variables import TransformedDistribution

tfb = tf.contrib.distributions.bijectors


def transform(x, *args, **kwargs):
  """Transform a continuous random variable to the unconstrained space.
  `transform` selects among a number of default transformations which
  depend on the support of the provided random variable:
  + $[0, 1]$ (e.g., Beta): Inverse of sigmoid.
  + $[0, \infty)$ (e.g., Gamma): Inverse of softplus.
  + Simplex (e.g., Dirichlet): Inverse of softmax-centered.
  + $(-\infty, \infty)$ (e.g., Normal, MultivariateNormalTriL): None.
  Args:
    x: RandomVariable.
      Continuous random variable to transform.
    *args, **kwargs:
      Arguments to overwrite when forming the `TransformedDistribution`.
      For example, manually specify the transformation by passing in
      the `bijector` argument.
  Returns:
    RandomVariable.
    A `TransformedDistribution` random variable, or the provided random
    variable if no transformation was applied.
  #### Examples
  ```python
  x = Gamma(1.0, 1.0)
  y = ed.transform(x)
  sess = tf.Session()
  sess.run(y)
  -2.2279539
  ```
  """
  if len(args) != 0 or kwargs.get('bijector', None) is not None:
    return TransformedDistribution(x, *args, **kwargs)

  try:
    support = x.support
  except AttributeError as e:
    msg = """'{}' object has no 'support'
             so cannot be transformed.""".format(type(x).__name__)
    raise AttributeError(msg)

  if support == '01':
    bij = tfb.Invert(tfb.Sigmoid())
    new_support = 'real'
  elif support == 'nonnegative':
    bij = tfb.Invert(tfb.Softplus())
    new_support = 'real'
  elif support == 'simplex':
    bij = tfb.Invert(tfb.SoftmaxCentered(event_ndims=1))
    new_support = 'multivariate_real'
  elif support in ('real', 'multivariate_real'):
    return x
  else:
    msg = "'transform' does not handle supports of type '{}'".format(support)
    raise ValueError(msg)

  new_x = TransformedDistribution(x, bij, *args, **kwargs)
  new_x.support = new_support
  return new_x
  '''
import torch
from abc import ABC, abstractmethod, ABCMeta
import os
from tqdm import tqdm
import numpy as np


class Inference(ABCMeta):
    '''
    Avstract base class for all inference methods. All inference methods will inherit from this class and will thus share
    all common methods and attributes.


    '''

    def __init__(self, model_code):
        '''
        Creates an inference algorithm.

        :param model_code
            type: str
            description:  This will interact with pyfoppl to generate a class, to compile the probabilistic program
            generate a graph describing that model. This class contains many methods for extracting latent parameters
            and generating the lod_Pdf.


        '''
        from pyfo.pyfoppl.foppl import imports
        from pyfo.pyfoppl.foppl import compilers

        graph, expr = compilers.compile(model_code)
        self.model = graph.create_model(result_expr=expr)
        
        self._cont_latents = None if len(self.model.gen_cont_vars())==0 else self.model.gen_cont_vars
        self._disc_latents = None if len(self.model.gen_disc_vars())==0 else self.model.gen_disc_vars
        self._if_latents =   None if len(self.model.gen_if_vars())==0 else self.model.gen_if_vars
        # each predicate has a cond boolean parameter, which represents the evaluated value of the predicate, whenever
        # the log_pdf is calculated.
        self._cond_bools = None if len(self.model.gen_if_vars()) ==0 else self.model.gen_cond_vars
        # all vertices of the graph are objects.
        self._vertices = self.model.get_vertices()
        self.observables = dict([ (vertex.name, vertex.observation) for vertex in self._vertices if vertex.is_observed])
        # a list of all vars
        self._all_vars =  [vertex.name for vertex in self._vertices if vertex.is_sampled]

        # distribution type of each latent variable, used for the bijections and embeddings
        self._cont_dists = dict([ (vertex.name, vertex.distribution_name) for vertex in self._vertices if (vertex.is_continuous and vertex.name in self._all_vars)])
        self._disc_dists = dict([ (vertex.name, vertex.distribution_name) for vertex in self._vertices if (vertex.is_discrete and vertex.name in self._all_vars)])

        # discrete support sizes
        self._disc_support =dict([ (vertex.name, vertex.support_size) for vertex in self._vertices if vertex.is_discrete ])

        # original parameter names. Parameter names are transformed in the compiler to ensure uniqueness
        self._names = dict([ (vertex.name, vertex.original_name) for vertex in self._vertices if vertex.name in self._all_vars ])


    @abstractmethod
    def initialize(self, n_iters=1000, n_chains=1, n_print=None, scale=None,
                   auto_transform=True, debug=False):
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
        :param n_print:
            type: int
            description: Number of iterations for each print progress. To suppress print
            progress, then specify 0. Default is `int(n_iter / 100)`.
        :param scale:
        :param auto_transform:
            type: bool
            description: Whether to automatically transform continuous latent variables
            of unequal support to be on the unconstrained space. It is
            only applied if the argument is `True`, the latent variable
            pair are `torch`s with the `support` attribute,
            the supports are both continuous and unequal.
        :param debug:
            type: bool
            If true, prints out graphical model.
            TODO : Link with a tensorboard style framework
        '''

        self.n_iters = n_iters
        if n_print is None:
            self.n_print = int(n_iters / 100)
        else:
            n_print

        # map from original latent vars to unconstrained vars
        self.transformations = {}

        if auto_transform:
            latent_vars =


        if debug:
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