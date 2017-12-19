import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{x30562 y30563}
	Arcs A:
	#{[x30562 y30563]}
	Conditional densities P:
	x30562 -> (fn [] (normal 1.0 5.0))
	y30563 -> (fn [x30562] (normal x30562 2.0))
	Observed values O:
	y30563 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x30562']

	@classmethod
	def gen_cont_vars(self):
		return ['x30562']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30586 = dist.Normal(mu=1.0, sigma=5.0)
		x30562 = dist30586.sample()   #sample 
		dist30588 = dist.Normal(mu=x30562, sigma=2.0)
		y30563 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30586 = dist.Normal(mu=1.0, sigma=5.0)
		x30562 =  state['x30562']   # get the x from input arg
		p30587 = dist30586.logpdf( x30562) # from prior
		dist30588 = dist.Normal(mu=x30562, sigma=2.0)
		y30563 = 7.0 
		p30589 = dist30588.logpdf(y30563) # from observe  
		logp =  p30587 + p30589  # total log joint 
		return logp # need to modify output format

