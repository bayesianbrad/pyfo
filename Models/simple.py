import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y32998 x32997}
	Arcs A:
	#{[x32997 y32998]}
	Conditional densities P:
	x32997 -> (fn [] (normal 1.0 5.0))
	y32998 -> (fn [x32997] (normal x32997 2.0))
	Observed values O:
	y32998 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x32997']

	@classmethod
	def gen_cont_vars(self):
		return ['x32997']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33021 = dist.Normal(mu=1.0, sigma=5.0)
		x32997 = dist33021.sample()   #sample 
		dist33023 = dist.Normal(mu=x32997, sigma=2.0)
		y32998 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33021 = dist.Normal(mu=1.0, sigma=5.0)
		x32997 =  state['x32997']   # get the x from input arg
		p33022 = dist33021.log_pdf( x32997) # from prior
		dist33023 = dist.Normal(mu=x32997, sigma=2.0)
		y32998 = 7.0 
		p33024 = dist33023.log_pdf(y32998) # from observe  
		logp =  p33022 + p33024  # total log joint 
		return logp # need to modify output format

