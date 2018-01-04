import torch
from torch.autograd import Variable
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
    @classmethod
    def gen_vars(self):
        return ['x30321']

    @classmethod
    def gen_cont_vars(self):
        return []

    @classmethod
    def gen_if_vars(self):
        return []

    @classmethod
    def gen_disc_vars(self):
        return ['x30321']

    # prior samples
    @classmethod
    def gen_prior_samples(self):
        dist30359 = dist.Normal(mean=0, std=1)
        x30321 = dist30359.sample()   #sample
        x30361 = logical_trans( x30321 > 0)
        dist30362 = dist.Normal(mean=1, std=1)
        y30324 = 1
        x30364 = logical_trans( x30321 > 0)
        x30365 = not logical_trans(x30364)
        dist30366 = dist.Normal(mean=-1, std=1)
        y30327 = 1
        state = {}
        for _gv in self.gen_vars():
            state[_gv] = locals()[_gv]
        return state # dictionary

    # compute pdf
    @classmethod
    def gen_pdf(self, state):
        dist30359 = dist.Normal(mean=0, std=1)
        x30321 =  state['x30321']    # get the x from input arg
        p30360 = dist30359.logpdf( x30321) # from prior
        x30361 = logical_trans( x30321 > 0)
        dist30362 = dist.Normal(mean=1, std=1)
        y30324 = 1
        p30363 = dist30362.logpdf( y30324) if x30361 else 0 # from observe with if
        x30364 = logical_trans( x30321 > 0)
        x30365 = not logical_trans(x30364)
        dist30366 = dist.Normal(mean=-1, std=1)
        y30327 = 1
        p30367 = dist30366.logpdf( y30327) if x30365 else 0 # from observe with if
        logp =  p30360 + p30363 + p30367  # total log joint
        return logp # need to m