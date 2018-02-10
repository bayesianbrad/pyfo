import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y33541 x33540}
	Arcs A:
	#{[x33540 y33541]}
	Conditional densities P:
	x33540 -> (fn [] (normal 1.0 5.0))
	y33541 -> (fn [x33540] (normal x33540 2.0))
	Observed values O:
	y33541 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x33540']

	@classmethod
	def gen_cont_vars(self):
		return ['x33540']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y33541', 'x33540']

	@classmethod
	def get_arcs(self):
		return [('x33540', 'y33541')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33564 = dist.Normal(mu=1.0, sigma=5.0)
		x33540 = dist33564.sample()   #sample 
		dist33566 = dist.Normal(mu=x33540, sigma=2.0)
		y33541 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33564 = dist.Normal(mu=1.0, sigma=5.0)
		x33540 =  state['x33540']   # get the x from input arg
		p33565 = dist33564.log_pdf( x33540) # from prior
		dist33566 = dist.Normal(mu=x33540, sigma=2.0)
		y33541 = 7.0 
		p33567 = dist33566.log_pdf(y33541) # from observe  
		logp =  p33565 + p33567  # total log joint 
		return logp # need to modify output format

