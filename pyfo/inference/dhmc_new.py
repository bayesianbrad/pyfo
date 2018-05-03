#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  17:49
Date created:  05/04/2018

License: MIT
'''
import torch
import copy
import math
import torch.distributions as dist
from pyfo.utils.unembed import Unembed
from pyfo.inference.hmc import HMC
from pyfo.utils.core import _to_leaf, VariableCast
class dhmc(HMC):

    def __init__(self, support_size):

        self._unembed_state = Unembed(support_size)
        super(HMC,self).__init__()

    def __generate_log_pdf(self, state, set_leafs=False, unembed=False, partial_unembed=False, key=None):
        """
        The compiled pytorch function, log_pdf, should automatically
        return the pdf.
        :param keys type: list of discrete embedded discrete parameters
        :return: log_pdf

        Do I need to convert the variables within state, to requires grad = True
        here? Then they will be passed to gen_logpdf to create the differentiable logpdf
        . Answer: yes, because
        """

        if unembed:
            _temp = self._unembed(state)
            if isinstance(_temp, torch.tensor) and math.isinf(_temp.data[0]):
                return _temp
            else:
                state = _temp
        if partial_unembed:
            _temp = self._partial_unembed(state, key)
            if isinstance(_temp, torch.tensor) and math.isinf(_temp.data[0]):
                return _temp
            else:
                state[key] = _temp[key]


        return self._gen_logpdf(state, set_leafs=set_leafs)

    def _embed(self, state, disc_key):
        """

        :param state:
        :param disc_key: Discrete parameter being embedded
        :return: embedded state with embedded parameter value

        else

        TO DO
        embedding : π(n)/ (a(n + 1/2) - a(n)) * I(a_{n-1/2} <  where a(n) = n n \mathbb{Z}^{+}
        This embedding essentially just divides by 1. Other embeddings for discrete parameters
         will be implemented later in the pipeline
        """
        embed_state = {}
        for key in disc_key:
            embed_state[key] = copy.copy(state[key])
        return embed_state

    def _partial_unembed(self, state, key):
        """
        Un-embeds only one parameter, to save on computation within the coordinate wise
        integrator.

        :param state:
        :param disc_key:
        :param feature to add later, regarding the support of the discrete param
        :return: state with unembedded value

        If embedded value falls outside of domain of discrete
        parameter return -inf
        simple embedding defined as:

         "Bernoulli", - support x \in {0,1}
        "Categorical", - x \in {0, \dots, k-1} where k \in \mathbb{Z}^{+}
        "Multinomial", - x_{i} \in {0,\dots ,n\}, i \in {0,\dots ,k-1} where sum(x_{i}) =n }
        "Poisson" - x_{i} \in {0,\dots ,+inf}  where x_{i} \in \mathbb{Z}^{+}

        TODO: ensure that the correct logpdf is being calculated
        """
        dist_name = 'unembed_' + self._disc_dist[key]
        unembed_var = getattr(self._unembed_state, dist_name)(state, key)
        return unembed_var

    def _unembed(self, state):
        """

        Un-embeds the entire state

        :param state:
        :param disc_key:
        :param feature to add later, regarding the support of the discrete param
        :return: state with unembedded value

        If embedded value falls outside of domain of discrete
        parameter return -inf
        simple embedding defined as:

         "Bernoulli", - support x \in {0,1}
        "Categorical", - x \in {0, \dots, k-1} where k \in \mathbb{Z}^{+}
        "Multinomial", - x_{i} \in {0,\dots ,n\}, i \in {0,\dots ,k-1} where sum(x_{i}) =n }
        "Poisson" - x_{i} \in {0,\dots ,+inf}  where x_{i} \in \mathbb{Z}^{+}

        """
        state_unembed = copy.copy(
            state)  # what is this doing? is it supposed to be passed as a arrgument instead of state?
        for key in self._disc_vars:
            dist_name = 'unembed_' + self._disc_dist[key]
            state = getattr(self._unembed_state, dist_name)(state, key)
        return state

    def momentum_sample(self, state):
        '''
        Generates either Laplacian or gaussian momentum dependent upon problem set-up

        :return:
        '''
        p = {}
        p_cont = dict([[key, dist.Normal(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._cont_latents]) if self._cont_latents is not None else {}
        p_disc = dict([[key, dist.Laplace(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._disc_latents]) if self._disc_latents is not None else {}
        p_if = dict([[key, dist.Laplace(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._if_latents]) if self._if_latents is not None else {}
        # quicker than using dict then update.
        p.update(p_cont)
        p.update(p_disc)
        p.update(p_if)
        return p

    def _energy(self, state, p, cont_keys):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param x:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :param cont keys: list of continous keys within state
        :return: Tensor
        """
        kinetic_disc = torch.sum(torch.stack([self.M * torch.dot(torch.abs(p[name]), torch.abs(p[name])) for name in self._disc_keys])) if self._disc_keys is not None else 0
        kinetic_cont = 0.5 * torch.sum(torch.stack([torch.dot(p[name], p[name]) for name in self._cont_keys])) if self._cont_keys is not None else 0
        kinetic_if = torch.sum(torch.stack([self.M * torch.dot(torch.abs(p[name]), torch.abs(p[name])) for name in self._if_keys])) if self._if_keys is not None else 0

        kinetic_energy = kinetic_cont + kinetic_disc + kinetic_if
        potential_energy = -self.log_posterior(state)

        return self._state._return_tensor(kinetic_energy) + self._state._return_tensor(potential_energy)
