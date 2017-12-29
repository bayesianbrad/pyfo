import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y31013 y31004 y31010 y31007 x31001}
	Arcs A:
	#{[x31001 y31004] [x31001 y31010] [x31001 y31013] [x31001 y31007]}
	Conditional densities P:
	x31001 -> (fn [] (normal 0 1))
	y31004 -> (fn [x31001] (if (and (< x31001 1) (> x31001 0)) (normal 0.5 1)))
	y31007 -> (fn [x31001] (if (and (not (< x31001 1)) (> x31001 0)) (normal 2 1)))
	y31010 -> (fn [x31001] (if (and (> x31001 -1) (not (> x31001 0))) (normal -0.5 1)))
	y31013 -> (fn [x31001] (if (and (not (> x31001 -1)) (not (> x31001 0))) (normal -2 1)))
	Observed values O:
	y31004 -> 1
	y31007 -> 1
	y31010 -> 1
	y31013 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x31001']

	@classmethod
	def gen_cont_vars(self):
		return ['x31001']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y31013', 'y31004', 'y31010', 'y31007', 'x31001']

	@classmethod
	def get_arcs(self):
		return [('x31001', 'y31004'), ('x31001', 'y31010'), ('x31001', 'y31013'), ('x31001', 'y31007')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist31058 = dist.Normal(mu=0, sigma=1)
		x31001 = dist31058.sample()   #sample 
		x31060 = logical_trans( x31001 < 1)
		x31061 = not logical_trans(x31060)
		x31062 = logical_trans( x31001 > 0)
		x31063 = logical_trans( x31061 and x31062)
		dist31064 = dist.Normal(mu=2, sigma=1)
		y31007 = 1 
		x31066 = logical_trans( x31001 > -1)
		x31067 = logical_trans( x31001 > 0)
		x31068 = not logical_trans(x31067)
		x31069 = logical_trans( x31066 and x31068)
		dist31070 = dist.Normal(mu=-0.5, sigma=1)
		y31010 = 1 
		x31072 = logical_trans( x31001 < 1)
		x31073 = logical_trans( x31001 > 0)
		x31074 = logical_trans( x31072 and x31073)
		dist31075 = dist.Normal(mu=0.5, sigma=1)
		y31004 = 1 
		x31077 = logical_trans( x31001 > -1)
		x31078 = not logical_trans(x31077)
		x31079 = logical_trans( x31001 > 0)
		x31080 = not logical_trans(x31079)
		x31081 = logical_trans( x31078 and x31080)
		dist31082 = dist.Normal(mu=-2, sigma=1)
		y31013 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist31058 = dist.Normal(mu=0, sigma=1)
		x31001 =  state['x31001']   # get the x from input arg
		p31059 = dist31058.log_pdf( x31001) # from prior
		x31060 = logical_trans( x31001 < 1)
		x31061 = not logical_trans(x31060)
		x31062 = logical_trans( x31001 > 0)
		x31063 = logical_trans( x31061 and x31062)
		dist31064 = dist.Normal(mu=2, sigma=1)
		y31007 = 1 
		p31065 = dist31064.(y31007) if x31063 else 0 # from observe with if  
		x31066 = logical_trans( x31001 > -1)
		x31067 = logical_trans( x31001 > 0)
		x31068 = not logical_trans(x31067)
		x31069 = logical_trans( x31066 and x31068)
		dist31070 = dist.Normal(mu=-0.5, sigma=1)
		y31010 = 1 
		p31071 = dist31070.(y31010) if x31069 else 0 # from observe with if  
		x31072 = logical_trans( x31001 < 1)
		x31073 = logical_trans( x31001 > 0)
		x31074 = logical_trans( x31072 and x31073)
		dist31075 = dist.Normal(mu=0.5, sigma=1)
		y31004 = 1 
		p31076 = dist31075.(y31004) if x31074 else 0 # from observe with if  
		x31077 = logical_trans( x31001 > -1)
		x31078 = not logical_trans(x31077)
		x31079 = logical_trans( x31001 > 0)
		x31080 = not logical_trans(x31079)
		x31081 = logical_trans( x31078 and x31080)
		dist31082 = dist.Normal(mu=-2, sigma=1)
		y31013 = 1 
		p31083 = dist31082.(y31013) if x31081 else 0 # from observe with if  
		logp =  p31059 + p31065 + p31071 + p31076 + p31083  # total log joint 
		return logp # need to modify output format

