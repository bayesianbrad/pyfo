import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x33079 y33080}
	Arcs A:
	#{[x33079 y33080]}
	Conditional densities P:
	x33079 -> (fn [] (normal 1.0 5.0))
	y33080 -> (fn [x33079] (normal (+ x33079 1) 2.0))
	Observed values O:
	y33080 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x33079']

	@classmethod
	def gen_cont_vars(self):
		return ['x33079']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x33079', 'y33080']

	@classmethod
	def get_arcs(self):
		return [('x33079', 'y33080')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33103 = dist.Normal(mu=1.0, sigma=5.0)
		x33079 = dist33103.sample()   #sample 
		x33105 =  x33079 + 1  
		dist33106 = dist.Normal(mu=x33105, sigma=2.0)
		y33080 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33103 = dist.Normal(mu=1.0, sigma=5.0)
		x33079 =  state['x33079']   # get the x from input arg
		p33104 = dist33103.log_pdf( x33079) # from prior
		x33105 =  x33079 + 1  
		dist33106 = dist.Normal(mu=x33105, sigma=2.0)
		y33080 = 7.0 
		p33107 = dist33106.log_pdf(y33080) # from observe  
		logp =  p33104 + p33107  # total log joint 
		return logp # need to modify output format

