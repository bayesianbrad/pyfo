#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:21
Date created:  17/01/2018

License: MIT
'''
import math
import torch
import decimal.Decimal as deci
from pyfo.utils.core import VariableCast
class Unembed():
    """

    Performs the unembedding map on the parameters who support
    needss transforming back to the original sample space.
    "Bernoulli", - support x \in {0,1}
    "Categorical", - x \in {0, \dots, k-1} where k \in \mathbb{Z}^{+}
    "Multinomial", - x_{i} \in {0,\dots ,n\}, i \in {0,\dots ,k-1} where sum(x_{i}) =n }
    "Poisson" - x_{i} \in {0,\dots ,+inf}  where x_{i} \in \mathbb{Z}^{+}

    """
    def __init__(self, disc_dist, dist_arg_size):
        self.dist_dist = disc_dist
        self.size = dist_arg_size

    def unembed_poisson(self, state,key):
        """
        unembed a poisson random variable

        :param state:
        :return:
        """
        lower = VariableCast(-0.5)
        if torch.lt(state[key].data, lower.data)

    def unembed_cat(self, state, key):
        """

        :param state:
        :return:
        """
        int_length = self.size[key]
        lower = VariableCast(-0.5)
        upper = VariableCast(int_length) + lower

        # Assumes each parameter represents 1-latent dimension
        if torch.gt(state[key].data,upper.data).data[0]:
            "outside region return -\inf"
            return -math.inf
        if torch.lt(state[key].data, lower.data).data[0]:
            return -math.inf
        else:
            state[key] = torch.ceil(state[key] + lower)
        return state

    def unembed_multino(self, state):
        """

        :param state:
        :return:
        """

    def unembed_binomial(self, state):
        """

        :param state:
        :return:
        """

    def to_decimal(self,float):
        return deci('%.2f' % float)