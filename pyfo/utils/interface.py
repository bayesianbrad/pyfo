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

        :return: List of continous random variables strings
        """
        raise NotImplementedError

    @classmethod
    def gen_disc_vars(self) -> List[str]:
        """

        :return: Lst of discrete random variables (RV from discrete distributions) strings
        """
        raise NotImplementedError
    @classmethod
    def gen_if_vars(self) -> List[str]:
        """

        :return: List of free random variables and its ancestors in if conditions as string
        """
        raise NotImplementedError
    @classmethod
    def gen_cond_vars(self) -> List[str]:
        """

        :return: List of free random variables and its ancestors in if conditions as string
        """
        raise NotImplementedError
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

    @classmethod
    def get_cond_functions(self) -> Dict[str,Variable]:
        """
        Returns a variable that represents the  of the predicate and the string representing which 'if'
        condition it represents

        Example:
        X1 ~ Normal(0,1)
        if (fX1 = X1 - a > 0):
            'do something'

        This if statement is has the name  'cond_1'

        :return:
        """

        raise NotImplementedError
    @classmethod
    def get_discrete_distribution(self) -> Dict[Variable,str]:
        """
        Returns the strings of the discrete parameters. I.e if x ~ Poission(1)
        then this is a dictionary of {x : 'Poisson'}
        :return:
        """
        raise NotImplementedError

    @classmethod
    def get_dist_parameter_size(self, name: str) -> tuple:
        """
        Used for understanding the size of the support, so that the unembedding is
        :param name: Represents discrete param name type: str
        :return: tuple of size (n,m)

        Example

        dist_sizes = {}
        if name in dist_sizes:
            return dist_sizes[name]
        else:
            return None


        """

        raise NotImplementedError
    @classmethod
    def get_vertices(self) -> List[str]:
        """
        Returns all vertices of the graphical model as a list of
        strings.

        :return vertices type: List[str]
        """
        raise NotImplementedError

    @classmethod
    def get_arcs(self) -> List[tuple]:
        """
        Returns all arcs/edges as tuples (u, v) with both u and v being string
        names of variables.

        :return arcs type: List[Tuple[str, str]]
        """
        raise NotImplementedError

    @classmethod
    def get_parents_map(self) -> Dict[str, set]:
        """
        Returns a dictionary of all variable names and the corresponding
        parents for each variable.

        Consider, for instance, a graph containing the following edges/arcs:
          x1 -> x3, x2 -> x3, x1 -> x4, x3 -> x4
        Then the returned value of this function is a dictionary as follow:
          x1: ()
          x2: ()
          x3: (x1, x2)
          x4: (x1, x3)
        In other words: you get a dictionary that maps each variable name to
        its correspondings parents.

        :return child_parent_relationships type: Dict[str, Set[str]]
        """
        result = { u: [] for u in self.get_vertices()}
        for (u, v) in self.get_arcs:
            if v in result:
                result[v].append(u)
            else:
                result[v] = [u]
        return { key: set(result[key]) for key in result }

    @classmethod
    def get_parents_of_node(self, var_name: str) -> set:
        """
        Returns a set of all variable names, which are parents of the given
        node/variable/vertex. This function basically extracts a single entry
        from the dictionary given by `get_parents_map`.

        :return set_of_parents :type Set[str]
        """
        edges = self.get_parents_map()
        if var_name in edges:
            return edges[var_name]
        else:
            return set()
