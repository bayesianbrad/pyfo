#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:53
Date created:  27/11/2017

License: MIT
'''

from typing import Dict, List, Bool, Set
import torch
from torch.autograd import Variable


class interface(object):
    """
    A generic model interface for DHMC
    """
    def __init__(self):

    def gen_vars() -> Set[str]:
        """
        Returns the names of the random variables in the model
        :param
        :return:
        """
        raise NotImplementedError

    def gen_cont_vars() -> Set[str]:
        """

        :return:
        """
        raise NotImplementedError

    def gen_disc_vars() -> Set[str]:
        """

        :return:
        """
        raise NotImplementedError

    # prior samples
    def gen_prior_samples() -> Dict[str,Variable]:
        """
        Returns a Dictionary whose entries are the string variable names and
        whose values are the sampled values for the same

        Generates a sample from the prior of all latent variables
        :return: Dict of sampled values
        """

        raise NotImplementedError

    # compute pdf
    def gen_pdf(Xs: Dict[str,Variable]) -> Variable:
        """
        Returns the log pdf of the model.

        :param Xs: Current values of the latent variables
        :return logp type: Variable
        """

        raise NotImplementedError

