import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{x30243 y30244}
	Arcs A:
	#{[x30243 y30244]}
	Conditional densities P:
	x30243 -> (fn [] (normal 1.0 5.0))
	y30244 -> (fn [x30243] (normal (+ x30243 1) 2.0))
	Observed values O:
	y30244 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x30243']

	@classmethod
	def gen_cont_vars(self):
		return ['x30243']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30267 = dist.Normal(mu=1.0, sigma=5.0)
		x30243 = dist30267.sample()   #sample 
		x30269 =  x30243 + 1  
		dist30270 = dist.Normal(mu=x30269, sigma=2.0)
		y30244 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30267 = dist.Normal(mu=1.0, sigma=5.0)
		x30243 =  state['x30243']   # get the x from input arg
		p30268 = dist30267.logpdf( x30243) # from prior
		x30269 =  x30243 + 1  
		dist30270 = dist.Normal(mu=x30269, sigma=2.0)
		y30244 = 7.0 
		p30271 = dist30270.logpdf(y30244) # from observe  
		logp =  p30268 + p30271  # total log joint 
		return logp # need to modify output format

