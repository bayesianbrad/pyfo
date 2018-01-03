import torch
from torch.autograd import Variable
import pyfo.distributions as dist
from pyfo.utils.interface import interface
from pyfo.utils.core import VariableCast

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
        dist30359 = dist.Normal(mu=0, sigma=1)
        x30321 = dist30359.sample()   #sample
        # x30361 = logical_trans( x30321 > 0)
        dist30362 = dist.Normal(mu=1, sigma=1)
        y30324 = 1
        dist30366 = dist.Normal(mu=-1, sigma=1)
        y30327 = 1
        state = {}
        for _gv in self.gen_vars():
            state[_gv] = locals()[_gv]
        return state # dictionary

    # compute pdf
    @classmethod
    def gen_pdf(self, state):
        dist30359 = dist.Normal(mu=0, sigma=1)
        x30321 =  state['x30321']    # get the x from input arg
        p30360 = dist30359.log_pdf( x30321) # from prior
        dist30362 = dist.Normal(mu=1, sigma=1)
        y30324 = 1
        dist30366 = dist.Normal(mu=-1, sigma=1)
        y30327 = 1
        if (x30321>0).data[0]:
            p30363 = dist30362.log_pdf( y30324) # from observe with if
            p30367 = VariableCast(0, grad=True)
        else:
            p30367 = dist30366.log_pdf(y30327)
            p30363 = VariableCast(0, grad=True)
        logp =  p30360 + p30363 + p30367  # total log joint
        return logp # need to m


# def logical_trans(var):
#    value = VariableCast(var)
#    if value.data[0]:
# 	   return True
#    else:
# 	   return False