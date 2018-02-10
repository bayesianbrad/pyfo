import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x33351 x33347 x33350}
	Arcs A:
	#{[x33350 x33351] [x33347 x33351]}
	Conditional densities P:
	x33347 -> (fn [] (poisson 2))
	x33350 -> (fn [] (poisson 7))
	x33351 -> (fn [x33347 x33350] (uniform-discrete x33347 x33350))
	Observed values O:
	
	'''

	@classmethod
	def gen_vars(self):
		return ['x33351', 'x33347', 'x33350']

	@classmethod
	def gen_cont_vars(self):
		return ['x33351', 'x33347', 'x33350']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x33351', 'x33347', 'x33350']

	@classmethod
	def get_arcs(self):
		return [('x33350', 'x33351'), ('x33347', 'x33351')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33380 = dist.Poisson(lam=7)
		x33350 = dist33380.sample()   #sample 
		dist33382 = dist.Poisson(lam=2)
		x33347 = dist33382.sample()   #sample 
		x33351 = .sample()   #sample 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33380 = dist.Poisson(lam=7)
		x33350 =  state['x33350']   # get the x from input arg
		p33381 = dist33380.log_pdf( x33350) # from prior
		dist33382 = dist.Poisson(lam=2)
		x33347 =  state['x33347']   # get the x from input arg
		p33383 = dist33382.log_pdf( x33347) # from prior
		x33351 =  state['x33351']   # get the x from input arg
		p33384 = .log_pdf( x33351) # from prior
		logp =  p33381 + p33383 + p33384  # total log joint 
		return logp # need to modify output format

