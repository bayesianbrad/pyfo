import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y32427 x32426}
	Arcs A:
	#{[x32426 y32427]}
	Conditional densities P:
	x32426 -> (fn [] (normal 1.0 5.0))
	y32427 -> (fn [x32426] (normal (+ x32426 1) 2.0))
	Observed values O:
	y32427 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x32426']

	@classmethod
	def gen_cont_vars(self):
		return ['x32426']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist32450 = dist.Normal(mu=1.0, sigma=5.0)
		x32426 = dist32450.sample()   #sample 
		x32452 =  x32426 + 1  
		dist32453 = dist.Normal(mu=x32452, sigma=2.0)
		y32427 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist32450 = dist.Normal(mu=1.0, sigma=5.0)
		x32426 =  state['x32426']   # get the x from input arg
		p32451 = dist32450.log_pdf( x32426) # from prior
		x32452 =  x32426 + 1  
		dist32453 = dist.Normal(mu=x32452, sigma=2.0)
		y32427 = 7.0 
		p32454 = dist32453.log_pdf(y32427) # from observe  
		logp =  p32451 + p32454  # total log joint 
		return logp # need to modify output format

