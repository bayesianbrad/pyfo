import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{y23094 x23093}
	Arcs A:
	#{[x23093 y23094]}
	Conditional densities P:
	x23093 -> (fn [] (normal 1.0 5.0))
	y23094 -> (fn [x23093] (normal (+ x23093 1) 2.0))
	Observed values O:
	y23094 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x23093']

	@classmethod
	def gen_cont_vars(self):
		return ['x23093']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist23119 = dist.Normal(mu=1.0, sigma=5.0)
		x23093 = dist23119.sample()   #sample 
		x23121 =  x23093 + 1  
		dist23122 = dist.Normal(mu=x23121, sigma=2.0)
		y23094 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist23119 = dist.Normal(mu=1.0, sigma=5.0)
		x23093 =  state['x23093']   # get the x from input arg
		p23120 = dist23119.logpdf( x23093) # from prior
		x23121 =  x23093 + 1  
		dist23122 = dist.Normal(mu=x23121, sigma=2.0)
		y23094 = 7.0 
		p23123 = dist23122.logpdf(y23094) # from observe  
		logp =  p23120 + p23123  # total log joint 
		return logp # need to modify output format

