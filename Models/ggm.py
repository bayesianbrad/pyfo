import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y32613 x32606}
	Arcs A:
	#{}
	Conditional densities P:
	x32606 -> (fn [] (categorical [0.3 0.7]))
	y32613 -> (fn [] (normal nil 2))
	Observed values O:
	y32613 -> 
	'''

	@classmethod
	def gen_vars(self):
		return ['x32606']

	@classmethod
	def gen_cont_vars(self):
		return ['x32606']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x32640 = [0.3,0.7]
		dist32641 = dist.Categorical(ps=x32640)
		x32606 = dist32641.sample()   #sample 
		dist32643 = dist.Normal(mu=, sigma=2)
		y32613 =  
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x32640 = [0.3,0.7]
		dist32641 = dist.Categorical(ps=x32640)
		x32606 =  state['x32606']   # get the x from input arg
		p32642 = dist32641.log_pdf( x32606) # from prior
		dist32643 = dist.Normal(mu=, sigma=2)
		y32613 =  
		p32644 = dist32643.log_pdf(y32613) # from observe  
		logp =  p32642 + p32644  # total log joint 
		return logp # need to modify output format

