import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y30316 x30315}
	Arcs A:
	#{[x30315 y30316]}
	Conditional densities P:
	x30315 -> (fn [] (normal 1.0 5.0))
	y30316 -> (fn [x30315] (normal (+ x30315 1) 2.0))
	Observed values O:
	y30316 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x30315']

	@classmethod
	def gen_cont_vars(self):
		return ['x30315']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y30316', 'x30315']

	@classmethod
	def get_arcs(self):
		return [('x30315', 'y30316')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30339 = dist.Normal(mu=1.0, sigma=5.0)
		x30315 = dist30339.sample()   #sample 
		x30341 =  x30315 + 1  
		dist30342 = dist.Normal(mu=x30341, sigma=2.0)
		y30316 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30339 = dist.Normal(mu=1.0, sigma=5.0)
		x30315 =  state['x30315']   # get the x from input arg
		p30340 = dist30339.log_pdf( x30315) # from prior
		x30341 =  x30315 + 1  
		dist30342 = dist.Normal(mu=x30341, sigma=2.0)
		y30316 = 7.0 
		p30343 = dist30342.log_pdf(y30316) # from observe  
		logp =  p30340 + p30343  # total log joint 
		return logp # need to modify output format

