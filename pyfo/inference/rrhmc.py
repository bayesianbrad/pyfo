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
import numpy as np
import torch.distributions as dist
from torch.distributions.transforms import constraints
from ..utils.unembed import Unembed
from ..inference.hmc import HMC
from ..utils.core import _to_leaf, _grad_logp, _generate_log_pdf


class rrhmc(HMC):

    def __init__(self, model_code, step_size=None,  num_steps=None, adapt_step_size=True, trajectory_length=None, **kwargs):
        self.generate_latent_vars()
        self.initialize()
        if self._disc_latents == None:
            return True
        else:
            import warnings
            warnings.warn('The DHMC integrator should have be called intiially. '
                          'now transferring to the DHMC integrator.')
            self._no_discrete()
        self.model_code = model_code
        self.adapt_step_size = adapt_step_size
        self._accept_cnt = 0
        self.__dict__.update(kwargs)
        super(HMC,self).__init__()


    def momentum_sample(self, state):
        '''
        Generates either Laplacian or gaussian momentum dependent upon problem set-up

        :return:
        '''
        p = {}
        p_cont = dict([[key, torch.randn(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._cont_latents]) if self._cont_latents is not None else {}
        p_if = dict([[key, torch.randn(loc=torch.zeros(state[key].size()),scale=torch.ones(state[key].size())).sample()] for key in self._if_latents]) if self._if_latents is not None else {}
        # quicker than using dict then update.
        p.update(p_cont)
        p.update(p_if)
        return p

    def _kinetic_energy(self, p):
        """
        All params are initiated with gaussian momentum
        :param p:
        :return:
        """
        return 0.5 * torch.sum(torch.stack([torch.mm(p[key].t(), p[key]) for key in self._all_vars]))
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
            # calculate the gradient field of the surface.

        return grad_vec

    def _half_step(self, state,logp, p):
        for key in self._all_vars:
            p[key] = p[key] + 0.5 * self.step_size * _grad_logp(input=logp, parameters=state[key])
        return p

    def sample(self, state):
        '''
        :param nsamples type: int descript: Specifies how many samples you would like to generate.
        :param burnin: type: int descript: Specifies how many samples you would like to remove.
        :param chains :type: int descript: Specifies the number of chains.
        :param save_data :type bool descrip: Specifies whether to save data and return data, or just return.
        :param dirname :type: str descrip: Path to a directory, where data can be saved.
        :return:
        '''

        # automatically transform `state` to unconstrained space, if needed.
        self.step_size = np.random.uniform(0.05, 0.18)
        self.num_steps = int(np.random.uniform(10, 20))
        for key, transform in self.transforms.items():
            # print('Debug statement in HMC.sample \n Printing transform : {0} '.format(transform))
            if transform is not constraints.real:
                state[key] = transform(state[key])
            else:
                continue
        p0 = self.momentum_sample(state)
        energy_current = self._energy(state, p0)

        state_new, p_new, logp = self._rrhmc_step(state, p0)


        # apply Metropolis correction.
        energy_proposal = logp + self._kinetic_energy(p_new)
        delta_energy = energy_proposal - energy_current
        rand = torch.rand(1)
        accept = torch.lt(rand, (-delta_energy).exp()).byte().any().item()
        if accept:
            self._accept_cnt += 1
            state = state_new

        # Return the unconstrained values for `state` to the constrianed values for 'state'.
        for key, transform in self.transforms.items():
            if transform is constraints.real:
                continue
            else:
                state[key] = transform.inv(state[key])
        return state

    def _no_discrete(self):
        """
        Ensures that there are no fully discrete distributions.
        If there are emits a warning.
        :param state:
        :return:
        """
            #TODO: write function that takes the number of samples, burn in etc and transfers to DHMC instance.

    def _rrhmc_step(self, state, p0, state_prev=None):
        """ This integrator would have been selected if we have if statements and no
        other forms of discreteness. It has a built in check to ensure self._all_vars
        contains either continuous and if vars, or just if vars."""
        # keep copies of the state and momentum. As the discontinuity maybe crossed.
        p0_copy = copy.copy(p0)
        state_copy = copy.copy(state)
        # need to check the change in conditions between the proposed_state and
        # original state
        bool = self._check_for_discontinuities(state_new=state, state_prev=state_prev)

        for i in range(self.num_steps):
            logp, state = self._potential_energy(state, set_leafs=True)
            p_new = self.halfstep(state, p0)



        return 0

    def _check_for_discontinuities(self,state_new, state_prev):
        """
        Checks to see if any of the conitional statements evaluated the
        consequence path. If so, it returns those keys and the while loop
        is initiated.

        :param state:
        :return:
        """
    def _binary_line_search(self):
        return 0