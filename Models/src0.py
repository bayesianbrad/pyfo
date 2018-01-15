import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y33612 x33611}
	Arcs A:
	#{[x33611 y33612]}
	Conditional densities P:
	x33611 -> (fn [] (normal 1.0 5.0))
	y33612 -> (fn [x33611] (normal x33611 2.0))
	Observed values O:
	y33612 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x33611']

	@classmethod
	def gen_cont_vars(self):
		return ['x33611']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33635 = dist.Normal(mu=1.0, sigma=5.0)
		x33611 = dist33635.sample()   #sample 
		dist33637 = dist.Normal(mu=x33611, sigma=2.0)
		y33612 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33635 = dist.Normal(mu=1.0, sigma=5.0)
		x33611 =  state['x33611']   # get the x from input arg
		p33636 = dist33635.log_pdf( x33611) # from prior
		dist33637 = dist.Normal(mu=x33611, sigma=2.0)
		y33612 = 7.0 
		p33638 = dist33637.log_pdf(y33612) # from observe  
		logp =  p33636 + p33638  # total log joint 
		return logp # need to modify output format

