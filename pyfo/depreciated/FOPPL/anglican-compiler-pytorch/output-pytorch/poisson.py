import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x30587 x30586 x30583}
	Arcs A:
	#{[x30583 x30587] [x30586 x30587]}
	Conditional densities P:
	x30583 -> (fn [] (poisson 2))
	x30586 -> (fn [] (poisson 7))
	x30587 -> (fn [x30586 x30583] (uniform-discrete x30583 x30586))
	Observed values O:
	
	'''

	@classmethod
	def gen_vars(self):
		return ['x30587', 'x30586', 'x30583']

	@classmethod
	def gen_cont_vars(self):
		return ['x30587', 'x30586', 'x30583']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x30587', 'x30586', 'x30583']

	@classmethod
	def get_arcs(self):
		return [('x30583', 'x30587'), ('x30586', 'x30587')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30616 = dist.Poisson(lam=2)
		x30583 = dist30616.sample()   #sample 
		dist30618 = dist.Poisson(lam=7)
		x30586 = dist30618.sample()   #sample 
		x30587 = .sample()   #sample 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30616 = dist.Poisson(lam=2)
		x30583 =  state['x30583']   # get the x from input arg
		p30617 = dist30616.log_pdf( x30583) # from prior
		dist30618 = dist.Poisson(lam=7)
		x30586 =  state['x30586']   # get the x from input arg
		p30619 = dist30618.log_pdf( x30586) # from prior
		x30587 =  state['x30587']   # get the x from input arg
		p30620 = .log_pdf( x30587) # from prior
		logp =  p30617 + p30619 + p30620  # total log joint 
		return logp # need to modify output format

