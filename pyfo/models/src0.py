import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{y30527 x30526}
	Arcs A:
	#{[x30526 y30527]}
	Conditional densities P:
	x30526 -> (fn [] (normal 1.0 5.0))
	y30527 -> (fn [x30526] (normal x30526 2.0))
	Observed values O:
	y30527 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x30526']

	@classmethod
	def gen_cont_vars(self):
		return ['x30526']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30550 = dist.Normal(mu=1.0, sigma=5.0)
		x30526 = dist30550.sample()   #sample 
		dist30552 = dist.Normal(mu=x30526, sigma=2.0)
		y30527 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30550 = dist.Normal(mu=1.0, sigma=5.0)
		x30526 =  state['x30526']   # get the x from input arg
		p30551 = dist30550.logpdf( x30526) # from prior
		dist30552 = dist.Normal(mu=x30526, sigma=2.0)
		y30527 = 7.0 
		p30553 = dist30552.logpdf(y30527) # from observe  
		logp =  p30551 + p30553  # total log joint 
		return logp # need to modify output format

