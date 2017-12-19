import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{y30035 y30041 y30038 x30032 y30044}
	Arcs A:
	#{[x30032 y30038] [x30032 y30035] [x30032 y30041] [x30032 y30044]}
	Conditional densities P:
	x30032 -> (fn [] (normal 0 1))
	y30035 -> (fn [x30032] (if (and (< x30032 1) (> x30032 0)) (normal 0.5 1)))
	y30038 -> (fn [x30032] (if (and (not (< x30032 1)) (> x30032 0)) (normal 2 1)))
	y30041 -> (fn [x30032] (if (and (> x30032 -1) (not (> x30032 0))) (normal -0.5 1)))
	y30044 -> (fn [x30032] (if (and (not (> x30032 -1)) (not (> x30032 0))) (normal -2 1)))
	Observed values O:
	y30035 -> 1
	y30038 -> 1
	y30041 -> 1
	y30044 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x30032']

	@classmethod
	def gen_cont_vars(self):
		return ['x30032']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30089 = dist.Normal(mu=0, sigma=1)
		x30032 = dist30089.sample()   #sample 
		x30091 = logical_trans( x30032 > -1)
		x30092 = not logical_trans(x30091)
		x30093 = logical_trans( x30032 > 0)
		x30094 = not logical_trans(x30093)
		x30095 = logical_trans( x30092 and x30094)
		dist30096 = dist.Normal(mu=-2, sigma=1)
		y30044 = 1 
		x30098 = logical_trans( x30032 < 1)
		x30099 = not logical_trans(x30098)
		x30100 = logical_trans( x30032 > 0)
		x30101 = logical_trans( x30099 and x30100)
		dist30102 = dist.Normal(mu=2, sigma=1)
		y30038 = 1 
		x30104 = logical_trans( x30032 > -1)
		x30105 = logical_trans( x30032 > 0)
		x30106 = not logical_trans(x30105)
		x30107 = logical_trans( x30104 and x30106)
		dist30108 = dist.Normal(mu=-0.5, sigma=1)
		y30041 = 1 
		x30110 = logical_trans( x30032 < 1)
		x30111 = logical_trans( x30032 > 0)
		x30112 = logical_trans( x30110 and x30111)
		dist30113 = dist.Normal(mu=0.5, sigma=1)
		y30035 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30089 = dist.Normal(mu=0, sigma=1)
		x30032 =  state['x30032']   # get the x from input arg
		p30090 = dist30089.logpdf( x30032) # from prior
		x30091 = logical_trans( x30032 > -1)
		x30092 = not logical_trans(x30091)
		x30093 = logical_trans( x30032 > 0)
		x30094 = not logical_trans(x30093)
		x30095 = logical_trans( x30092 and x30094)
		dist30096 = dist.Normal(mu=-2, sigma=1)
		y30044 = 1 
		p30097 = dist30096.logpdf(y30044) if x30095 else 0 # from observe with if  
		x30098 = logical_trans( x30032 < 1)
		x30099 = not logical_trans(x30098)
		x30100 = logical_trans( x30032 > 0)
		x30101 = logical_trans( x30099 and x30100)
		dist30102 = dist.Normal(mu=2, sigma=1)
		y30038 = 1 
		p30103 = dist30102.logpdf(y30038) if x30101 else 0 # from observe with if  
		x30104 = logical_trans( x30032 > -1)
		x30105 = logical_trans( x30032 > 0)
		x30106 = not logical_trans(x30105)
		x30107 = logical_trans( x30104 and x30106)
		dist30108 = dist.Normal(mu=-0.5, sigma=1)
		y30041 = 1 
		p30109 = dist30108.logpdf(y30041) if x30107 else 0 # from observe with if  
		x30110 = logical_trans( x30032 < 1)
		x30111 = logical_trans( x30032 > 0)
		x30112 = logical_trans( x30110 and x30111)
		dist30113 = dist.Normal(mu=0.5, sigma=1)
		y30035 = 1 
		p30114 = dist30113.logpdf(y30035) if x30112 else 0 # from observe with if  
		logp =  p30090 + p30097 + p30103 + p30109 + p30114  # total log joint 
		return logp # need to modify output format

