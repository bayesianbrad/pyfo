import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x32690 x32693 x32694}
	Arcs A:
	#{[x32693 x32694] [x32690 x32694]}
	Conditional densities P:
	x32690 -> (fn [] (poisson 2))
	x32693 -> (fn [] (poisson 7))
	x32694 -> (fn [x32690 x32693] (uniform-discrete x32690 x32693))
	Observed values O:
	
	'''

	@classmethod
	def gen_vars(self):
		return ['x32690', 'x32693', 'x32694']

	@classmethod
	def gen_cont_vars(self):
		return ['x32690', 'x32693', 'x32694']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist32723 = dist.Poisson(lam=7)
		x32693 = dist32723.sample()   #sample 
		dist32725 = dist.Poisson(lam=2)
		x32690 = dist32725.sample()   #sample 
		x32694 = .sample()   #sample 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist32723 = dist.Poisson(lam=7)
		x32693 =  state['x32693']   # get the x from input arg
		p32724 = dist32723.log_pdf( x32693) # from prior
		dist32725 = dist.Poisson(lam=2)
		x32690 =  state['x32690']   # get the x from input arg
		p32726 = dist32725.log_pdf( x32690) # from prior
		x32694 =  state['x32694']   # get the x from input arg
		p32727 = .log_pdf( x32694) # from prior
		logp =  p32724 + p32726 + p32727  # total log joint 
		return logp # need to modify output format

