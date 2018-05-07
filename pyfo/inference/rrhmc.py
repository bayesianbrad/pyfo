#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  00:06
Date created:  07/05/2018

License: MIT
'''
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
from pyfo.utils.core import _to_leaf, _grad_logp, _generate_log_pdf

class rrhmc(HMC):

    def __init__(self, support_size):

        self._unembed_state = Unembed(support_size)
        super(HMC,self).__init__()

    def _potential_energy(self, state, set_leafs=False):
        """
        The compiled pytorch function, log_pdf, should automatically
        return the pdf.
        :param keys type: list of discrete embedded discrete parameters
        :return: log_pdf

        Do I need to convert the variables within state, to requires grad = True
        here? Then they will be passed to gen_logpdf to create the differentiable logpdf
        . Answer: yes, because
        """



        return self._gen_logpdf(state, set_leafs=set_leafs)


    def momentum_sample(self, state):
        '''
        Generates either Laplacian or gaussian momentum dependent upon problem set-up

        :return:
        '''
        p = {}
        p_cont = dict([[key, torch.randn(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._cont_latents]) if self._cont_latents is not None else {}
        p_disc = dict([[key, dist.Laplace(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._disc_latents]) if self._disc_latents is not None else {}
        p_if = dict([[key, dist.Laplace(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._if_latents]) if self._if_latents is not None else {}
        # quicker than using dict then update.
        p.update(p_cont)
        p.update(p_disc)
        p.update(p_if)
        return p

    def _kinetic_energy(self, p):
        """
        Calculates the discrete kinetic energy

        :param p:
        :return:
        """
        kinetic_disc = torch.sum(torch.stack([self.M * torch.dot(torch.abs(p[name]), torch.abs(p[name])) for name in self._disc_keys])) if self._disc_keys is not None else 0
        kinetic_cont = 0.5 * torch.sum(torch.stack([torch.dot(p[name], p[name]) for name in self._cont_keys])) if self._cont_keys is not None else 0
        kinetic_if = torch.sum(torch.stack([self.M * torch.dot(torch.abs(p[name]), torch.abs(p[name])) for name in self._if_keys])) if self._if_keys is not None else 0

        kinetic_energy = kinetic_cont + kinetic_disc + kinetic_if

    def _energy(self, state, p):
        """
        Calculates the hamiltonian for calculating the acceptance ration (detailed balance)
        :param state:  Dictionary of full program state
        :param p:  Dictionary of momentum
        :return: Tensor
        """

        return self._kinetic_energy(p=p) + self._potential_energy(state=state)

    def _gradient_field(self, key, state):
        """
        For RRHMC

        :param state: dict of all latent variables
        :param key: the value of predicate
        :return: unit grad Vector type: List of Variables
        Tobias has now created a dictionary of the predicates,
        that has for the value a lambda function that returns
        the scalar field .

        To DO: Finish once tobias gets back, as the function is including variables that
        do not exist in the outputted parameter fields. Will need to use ancestor to generate the keys for the history,
        so that we know what to differentiate with respect to
        """
        ancestors = self._ancestors(key)  # returns set
        scalar_field = []  # TODO Complete scalar field construction
        grad_vec = []
        for ancestor in ancestors:
            grad_vec.append(torch.autograd.grad(scalar_field, ancestor, retain_graph=True))

        return grad_vec

    def _rrhmc_integrator(self):
        return 0

    def _binary_line_search(self):
        return 0