import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{x30602 y30609}
	Arcs A:
	#{}
	Conditional densities P:
	x30602 -> (fn [] (categorical [0.3 0.7]))
	y30609 -> (fn [] (normal nil 2))
	Observed values O:
	y30609 -> 
	'''

	@classmethod
	def gen_vars(self):
		return ['x30602']

	@classmethod
	def gen_cont_vars(self):
		return ['x30602']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30636 = dist.Normal(mu=, sigma=2)
		y30609 =  
		x30638 = [0.3,0.7]
		dist30639 = dist.Categorical(p=x30638)
		x30602 = dist30639.sample()   #sample 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30636 = dist.Normal(mu=, sigma=2)
		y30609 =  
		p30637 = dist30636.logpdf(y30609) # from observe  
		x30638 = [0.3,0.7]
		dist30639 = dist.Categorical(p=x30638)
		x30602 =  state['x30602']   # get the x from input arg
		p30640 = dist30639.logpdf( x30602) # from prior
		logp =  p30637 + p30640  # total log joint 
		return logp # need to modify output format

