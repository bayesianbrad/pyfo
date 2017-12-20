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
    @classmethod
    def gen_vars(self) -> List[str]:
        """
        Returns the names of the random variables in the model
        :param
        :return:
        """
        raise NotImplementedError

    @classmethod
    def gen_cont_vars(self) -> List[str]:
        """

        :return: List of continous arameter strings
        """
        raise NotImplementedError

    @classmethod
    def gen_disc_vars(self) -> List[str]:
        """

        :return: Lst of discrete parameter strings
        """
        raise NotImplementedError
    @classmethod
    def gen_if_vars(self) -> List[str]:
        """

        :return: List of if ancestors as string
        """
    # prior samples
    @classmethod
    def gen_prior_samples(self) -> Dict[str,Variable]:
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
    @classmethod
    def gen_pdf(self, state: Dict[str,Variable]) -> Variable:
        """
        Returns the log pdf of the model.
        Pass stat with each variable as  a leaf node, so that the
        gradients accumalate throughout this method

        :param x: Current values of the latent variables
        :return logp type: Variable
        """

        raise NotImplementedError

    # need to know which if branch we have gone down and what