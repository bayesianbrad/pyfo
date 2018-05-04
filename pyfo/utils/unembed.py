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
from torch.autograd import Variable
import time
#TODO: Ensure the correctness of this.  
class Unembed():
    """

    Performs the unembedding map on the parameters who support
    needss transforming back to the original sample space.
    "Bernoulli", - support x \in {0,1}
    "Categorical", - x \in {0, \dots, k-1} where k \in \mathbb{Z}^{+}
    "Multinomial", - x_{i} \in {0,\dots ,n\}, i \in {0,\dots ,k-1} where sum(x_{i}) =n }
    "Poisson" - x_{i} \in {0,\dots ,+inf}  where x_{i} \in \mathbb{Z}^{+}

    You don't see the indicator functions below, as they are implicit.

    """
    def __init__(self, support_sizes):
        self._support_sizes = support_sizes

    def unembed_Poisson(self, state,key):
        """
        unembed a poisson random variable
        Original support {0,1,2,..., +\inf}
        Transformed support [0,+/inf)


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
        assumes that the categorical is embedded in the region \frac{n}{a_{n+1/2} - a_n{n - 1/2}} where -0.5<<...< n-1/2
        Original support was the set {0,1, ...., n}
        Transformed support (-0.5,n-1/2)
        :param state:
        :return:
        """
        # print('Printing self._support_sizes :', self._support_sizes)
        if not isinstance(state[key], Variable):
            state[key]= VariableCast(state[key])
        int_length = VariableCast(self._support_sizes[key]).expand(state[key].size())
        lower = VariableCast(0).expand(state[key].size())
        upper = int_length+ lower

        # # Assumes each parameter represents 1-latent dimension
        # print("Debug statement in unembed.unembed_Categorical \n"
        #       "The type of upper is: {0}  \n"
        #       "The type of state[{2}] is: {1} \n"
        #       "The value of state[{2}] is: {3} ".format(type(upper), type(state[key]), key, state[key]))
        if torch.gt(state[key],upper).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        elif torch.lt(state[key], lower).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        # if torch.le(state[key],upper + 2*lower).data[0] and torch.le(state[key],upper).data[0] :
        #     state[key] = torch.round(state[key]) #equiv to torch.round(upper)
        else:
            state[key] = torch.floor(state[key])
        return state

    def unembed_Multinomial(self, state):
        """

        :param state:
        :return:
        """
        raise NotImplementedError

    def unembed_Binomial(self, state,key):
        """
        umembeds the binomial random variable
        Original support {0,1, ....,n}
        Transformed support (0, n)

        :param state:
        :return:
        """
        lower = VariableCast(0).expand(state[key].size())
        upper = VariableCast(self._support_sizes[key]).expand(state[key].size()) - lower
        if torch.lt(state[key], lower).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        if torch.gt(state[key], upper).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        if torch.lt(state[key], upper).data[0] and torch.gt(state[key], upper + 2 * lower).data[0]:
            state[key] = VariableCast(torch.round(state[key]))  # equiv to torch.round(upper)
        else:
            state[key] = VariableCast(torch.round(state[key] - lower))
        return state

    def unembed_Bernoulli(self, state, key):
        """
              umembeds the binomial random variable
              Original support {0,1}
              Transformed support [0, \inf)

              :param state:
              :return:
        """
        lower = VariableCast(0).expand(state[key].size())
        upper = VariableCast(self._support_sizes[key]).expand(state[key].size()) - lower

        # # Assumes each parameter represents 1-latent dimension
        # print("Debug statement in unembed.unembed_Categorical \n"
        #       "The type of upper is: {0}  \n"
        #       "The type of state[{2}] is: {1} \n"
        #       "The value of state[{2}] is: {3} ".format(type(upper), type(state[key]), key, state[key]))
        if not isinstance(state[key], Variable):
            state[key] = VariableCast(state[key])
        if torch.lt(state[key], lower).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        if torch.gt(state[key], upper).data[0]:
            "outside region return -\inf"
            return VariableCast(-math.inf)
        if torch.lt(state[key], upper).data[0] and torch.gt(state[key], upper + 2 * lower).data[0]:
            state[key] = VariableCast(torch.round(state[key]))  # equiv to torch.round(upper)
        else:
            state[key] = VariableCast(torch.round(state[key] - lower))
        return state
