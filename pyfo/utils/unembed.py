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
    def __init__(self, dist_arg_size):
        self.size = dist_arg_size

    def unembed_Poisson(self, state,key):
        """
        unembed a poisson random variable

        :param state:
        :return:
        """
        lower = VariableCast(0)
        if torch.lt(state[key], lower).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        else:
            state[key] = torch.floor(state[key])
        return state

    def unembed_Categorical(self, state, key):
        """

        :param state:
        :return:
        """
        # print('Debug statement in Unembed.unembed.unembed_Categorical\n'
        #       'Print self.size of disc parameter : {0} '.format(self.size(key)))
        int_length = self.size(key)[0]
        lower = VariableCast(-0.5)
        upper = VariableCast(int_length) + lower

        # # Assumes each parameter represents 1-latent dimension
        # print("Debug statement in unembed.unembed_Categorical \n"
        #       "The type of upper is: {0}  \n"
        #       "The type of state[{2}] is: {1} \n"
        #       "The value of state[{2}] is: {3} ".format(type(upper), type(state[key]), key, state[key]))
        if torch.gt(state[key],upper).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        if torch.lt(state[key], lower).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        if torch.lt(state[key],upper).data[0] and torch.gt(state[key],upper + 2*lower).data[0]:
            state[key] = VariableCast(torch.round(state[key])) #equiv to torch.round(upper)
        else:
            state[key] = VariableCast(torch.round(state[key] - lower))
        return state

    def unembed_Multinomial(self, state):
        """

        :param state:
        :return:
        """
        raise NotImplementedError

    def unembed_Binomial(self, state):
        """

        :param state:
        :return:
        """
        raise NotImplementedError