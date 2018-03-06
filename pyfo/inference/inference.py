#!/usr/bin/env python3
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

        :param model
            type: class
            description: The model model code. This will interact with pyfoppl to generate a class, which contains

        '''
        self.model = self.model
    @abstractmethod
    def initialize(self, n_iters=1000, n_chains=1, n_print=None, scale=None,
                 auto_transform=True, debug=False):
        '''
        Initialize inference algorithm. It initializes hyperparameters
        and builds ops for the algorithm's computation graph.

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
            self.n_print = int(n_iters/100)
        else:
            n_print

        self.indices = np.random.randint(0, n_chians * len_trace, n_iters)


