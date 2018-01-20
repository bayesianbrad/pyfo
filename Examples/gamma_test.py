#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  14:28
Date created:  13/01/2018

License: MIT
'''
import math
import numpy as np
import torch
from torch.autograd import Variable
import pyfo.distributions as dist
from pyfo.utils.interface import interface


class model(interface):
    """
    Vertices V:
      x20001
    Arcs A:

    Conditional densities C:
      x20001 -> dist.Gamma(alpha=3.0, beta=4.0)
    Observed values O:

    """
    vertices = {'x20001'}
    arcs = set()
    names = {'x20001': 'x1'}
    disc_params = {'x20001': 'Poisson'}
    cond_functions = {

    }

    @classmethod
    def get_vertices(self):
        return list(self.vertices)

    @classmethod
    def get_arcs(self):
        return list(self.arcs)

    @classmethod
    def gen_cond_vars(self):
        return []

    @classmethod
    def gen_cont_vars(self):
        return ['x20001']

    @classmethod
    def gen_disc_vars(self):
        return []

    @classmethod
    def gen_if_vars(self):
        return []

    @classmethod
    def gen_pdf(self, state):
        dist_x20001 = dist.Gamma(alpha=3.0, beta=4.0)
        x20001 = state['x20001']
        p10000 = dist_x20001.log_pdf(x20001)
        logp = p10000
        return logp

    @classmethod
    def gen_prior_samples(self):
        dist_x20001 = dist.Gamma(alpha=3.0, beta=4.0)
        x20001 = dist_x20001.sample()
        state = {}
        for _gv in self.gen_vars():
            state[_gv] = locals()[_gv]
        return state  # dictionary

    @classmethod
    def gen_vars(self):
        return ['x20001']