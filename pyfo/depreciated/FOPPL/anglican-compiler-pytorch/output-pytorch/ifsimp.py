import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x31095 x31092 y31099 y31096}
	Arcs A:
	#{[x31092 y31096] [x31092 y31099] [x31095 y31096]}
	Conditional densities P:
	x31092 -> (fn [] (normal 0 1))
	x31095 -> (fn [] (normal 0 1))
	y31096 -> (fn [x31095 x31092] (if (> x31092 0) (normal x31095 1)))
	y31099 -> (fn [x31092] (if (not (> x31092 0)) (normal -1 1)))
	Observed values O:
	y31096 -> 1
	y31099 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x31095', 'x31092']

	@classmethod
	def gen_cont_vars(self):
		return ['x31095', 'x31092']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x31095', 'x31092', 'y31099', 'y31096']

	@classmethod
	def get_arcs(self):
		return [('x31092', 'y31096'), ('x31092', 'y31099'), ('x31095', 'y31096')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist31134 = dist.Normal(mu=0, sigma=1)
		x31092 = dist31134.sample()   #sample 
		x31136 = logical_trans( x31092 > 0)
		x31137 = not logical_trans(x31136)
		dist31138 = dist.Normal(mu=-1, sigma=1)
		y31099 = 1 
		dist31140 = dist.Normal(mu=0, sigma=1)
		x31095 = dist31140.sample()   #sample 
		x31142 = logical_trans( x31092 > 0)
		dist31143 = dist.Normal(mu=x31095, sigma=1)
		y31096 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist31134 = dist.Normal(mu=0, sigma=1)
		x31092 =  state['x31092']   # get the x from input arg
		p31135 = dist31134.log_pdf( x31092) # from prior
		x31136 = logical_trans( x31092 > 0)
		x31137 = not logical_trans(x31136)
		dist31138 = dist.Normal(mu=-1, sigma=1)
		y31099 = 1 
		p31139 = dist31138.(y31099) if x31137 else 0 # from observe with if  
		dist31140 = dist.Normal(mu=0, sigma=1)
		x31095 =  state['x31095']   # get the x from input arg
		p31141 = dist31140.log_pdf( x31095) # from prior
		x31142 = logical_trans( x31092 > 0)
		dist31143 = dist.Normal(mu=x31095, sigma=1)
		y31096 = 1 
		p31144 = dist31143.(y31096) if x31142 else 0 # from observe with if  
		logp =  p31135 + p31139 + p31141 + p31144  # total log joint 
		return logp # need to modify output format

