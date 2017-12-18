#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  15:53
Date created:  27/11/2017

License: MIT
'''

from typing import Dict, List
from torch.autograd import Variable


class interface():
    """
    A generic model interface for DHMC
    """
    @staticmethod
    def gen_vars() -> List[str]:
        """
        Returns the names of the random variables in the model
        :param
        :return:
        """
    raise NotImplementedError

    @staticmethod
    def gen_cont_vars() -> List[str]:
        """

        :return:
        """
        raise NotImplementedError

    @staticmethod
    def gen_disc_vars() -> List[str]:
        """

        :return:
        """
        raise NotImplementedError

    # prior samples
    @staticmethod
    def gen_prior_samples() -> Dict[str,Variable]:
        """
        Returns a Dictionary whose entries are the string variable names and
        whose values are the sampled values for the same

        Generates a sample from the prior of all latent variables
        :return: Dict of sampled values
        """
        # map (bitmap) from every if statement. A list of if statements encontered
        #    and whether or not we went down the consequent or alternative branch

        raise NotImplementedError

    # compute pdf
    @staticmethod
    def gen_pdf(x: Dict[str,Variable]) -> Variable:
        """
        Returns the log pdf of the model.
        Pass stat with each variable as  a leaf node, so that the
        gradients accumalate throughout this method

        :param x: Current values of the latent variables
        :return logp type: Variable
        """

        raise NotImplementedError

    # need to know which if branch we have gone down and what