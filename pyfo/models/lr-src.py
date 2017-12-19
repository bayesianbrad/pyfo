import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{y30422 y30401 x30394 y30447 x30391}
	Arcs A:
	#{[x30394 y30422] [x30391 y30401] [x30394 y30401] [x30391 y30422] [x30391 y30447] [x30394 y30447]}
	Conditional densities P:
	x30391 -> (fn [] (normal 0.0 10.0))
	x30394 -> (fn [] (normal 0.0 10.0))
	y30401 -> (fn [x30394 x30391] (normal (+ (* x30391 1.0) x30394) 1.0))
	y30422 -> (fn [x30394 x30391] (normal (+ (* x30391 2.0) x30394) 1.0))
	y30447 -> (fn [x30394 x30391] (normal (+ (* x30391 3.0) x30394) 1.0))
	Observed values O:
	y30401 -> 2.1
	y30422 -> 3.9
	y30447 -> 5.3
	'''

	@classmethod
	def gen_vars(self):
		return ['x30394', 'x30391']

	@classmethod
	def gen_cont_vars(self):
		return ['x30394', 'x30391']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30502 = dist.Normal(mu=0.0, sigma=10.0)
		x30391 = dist30502.sample()   #sample 
		dist30504 = dist.Normal(mu=0.0, sigma=10.0)
		x30394 = dist30504.sample()   #sample 
		x30506 = x30391 * 3.0
		x30507 =  x30506 + x30394  
		dist30508 = dist.Normal(mu=x30507, sigma=1.0)
		y30447 = 5.3 
		x30510 = x30391 * 1.0
		x30511 =  x30510 + x30394  
		dist30512 = dist.Normal(mu=x30511, sigma=1.0)
		y30401 = 2.1 
		x30514 = x30391 * 2.0
		x30515 =  x30514 + x30394  
		dist30516 = dist.Normal(mu=x30515, sigma=1.0)
		y30422 = 3.9 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30502 = dist.Normal(mu=0.0, sigma=10.0)
		x30391 =  state['x30391']   # get the x from input arg
		p30503 = dist30502.logpdf( x30391) # from prior
		dist30504 = dist.Normal(mu=0.0, sigma=10.0)
		x30394 =  state['x30394']   # get the x from input arg
		p30505 = dist30504.logpdf( x30394) # from prior
		x30506 = x30391 * 3.0
		x30507 =  x30506 + x30394  
		dist30508 = dist.Normal(mu=x30507, sigma=1.0)
		y30447 = 5.3 
		p30509 = dist30508.logpdf(y30447) # from observe  
		x30510 = x30391 * 1.0
		x30511 =  x30510 + x30394  
		dist30512 = dist.Normal(mu=x30511, sigma=1.0)
		y30401 = 2.1 
		p30513 = dist30512.logpdf(y30401) # from observe  
		x30514 = x30391 * 2.0
		x30515 =  x30514 + x30394  
		dist30516 = dist.Normal(mu=x30515, sigma=1.0)
		y30422 = 3.9 
		p30517 = dist30516.logpdf(y30422) # from observe  
		logp =  p30503 + p30505 + p30509 + p30513 + p30517  # total log joint 
		return logp # need to modify output format

