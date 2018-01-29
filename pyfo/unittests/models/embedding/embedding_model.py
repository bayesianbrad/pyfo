import torch
from torch.autograd import Variable
import pyfo.distributions as dist
from pyfo.utils.interface import interface
from pyfo.utils.core import VariableCast

class model(interface):
	@classmethod
	def gen_vars(self):
		return ['x37011', 'x37012']

	@classmethod
	def gen_cont_vars(self):
		return []

	@classmethod
	def gen_if_vars(self):
		return []

	@classmethod
	def gen_disc_vars(self):
		return ['x37011', 'x37012']

	# prior samples
	@classmethod
	def gen_prior_samples(self):
		x37042 = [0.1,0.2,0.7]
		dist37043 = dist.Categorical(ps=x37042)
		x37011 = dist37043.sample()   #sample
		dist37045 = dist.Poisson(lam=x37011)
		x37012 = dist37045.sample()   #sample
		dist37047 = dist.Normal(mu=x37012, sigma=2)
		y37013 = 1
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf
	@classmethod
	def gen_pdf(self, state):
		x37042 = [0.1,0.2,0.7]
		dist37043 = dist.Categorical(ps=x37042)
		x37011 =  state['x37011']    # get the x from input arg
		p37044 = dist37043.logpdf( x37011) # from prior
		dist37045 = dist.Poisson(lam=x37011)
		x37012 =  state['x37012']    # get the x from input arg
		p37046 = dist37045.logpdf( x37012) # from prior
		dist37047 = dist.Normal(mu=x37012, sigma=2)
		y37013 = 1
		p37048 = dist37047.logpdf( y37013) # from observe
		logp =  p37044 + p37046 + p37048  # total log joint
		return logp # need to modify output format

