import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y30913 y30919 y30922 x30910 y30916}
	Arcs A:
	#{[x30910 y30916] [x30910 y30919] [x30910 y30913] [x30910 y30922]}
	Conditional densities P:
	x30910 -> (fn [] (normal 0 1))
	y30913 -> (fn [x30910] (if (and (< x30910 1) (> x30910 0)) (normal 0.5 1)))
	y30916 -> (fn [x30910] (if (and (not (< x30910 1)) (> x30910 0)) (normal 2 1)))
	y30919 -> (fn [x30910] (if (and (> x30910 -1) (not (> x30910 0))) (normal -0.5 1)))
	y30922 -> (fn [x30910] (if (and (not (> x30910 -1)) (not (> x30910 0))) (normal -2 1)))
	Observed values O:
	y30913 -> 1
	y30916 -> 1
	y30919 -> 1
	y30922 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x30910']

	@classmethod
	def gen_cont_vars(self):
		return ['x30910']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y30913', 'y30919', 'y30922', 'x30910', 'y30916']

	@classmethod
	def get_arcs(self):
		return [('x30910', 'y30916'), ('x30910', 'y30919'), ('x30910', 'y30913'), ('x30910', 'y30922')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30967 = dist.Normal(mu=0, sigma=1)
		x30910 = dist30967.sample()   #sample 
		x30969 = logical_trans( x30910 < 1)
		x30970 = not logical_trans(x30969)
		x30971 = logical_trans( x30910 > 0)
		x30972 = logical_trans( x30970 and x30971)
		dist30973 = dist.Normal(mu=2, sigma=1)
		y30916 = 1 
		x30975 = logical_trans( x30910 > -1)
		x30976 = not logical_trans(x30975)
		x30977 = logical_trans( x30910 > 0)
		x30978 = not logical_trans(x30977)
		x30979 = logical_trans( x30976 and x30978)
		dist30980 = dist.Normal(mu=-2, sigma=1)
		y30922 = 1 
		x30982 = logical_trans( x30910 > -1)
		x30983 = logical_trans( x30910 > 0)
		x30984 = not logical_trans(x30983)
		x30985 = logical_trans( x30982 and x30984)
		dist30986 = dist.Normal(mu=-0.5, sigma=1)
		y30919 = 1 
		x30988 = logical_trans( x30910 < 1)
		x30989 = logical_trans( x30910 > 0)
		x30990 = logical_trans( x30988 and x30989)
		dist30991 = dist.Normal(mu=0.5, sigma=1)
		y30913 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30967 = dist.Normal(mu=0, sigma=1)
		x30910 =  state['x30910']   # get the x from input arg
		p30968 = dist30967.log_pdf( x30910) # from prior
		x30969 = logical_trans( x30910 < 1)
		x30970 = not logical_trans(x30969)
		x30971 = logical_trans( x30910 > 0)
		x30972 = logical_trans( x30970 and x30971)
		dist30973 = dist.Normal(mu=2, sigma=1)
		y30916 = 1 
		p30974 = dist30973.(y30916) if x30972 else 0 # from observe with if  
		x30975 = logical_trans( x30910 > -1)
		x30976 = not logical_trans(x30975)
		x30977 = logical_trans( x30910 > 0)
		x30978 = not logical_trans(x30977)
		x30979 = logical_trans( x30976 and x30978)
		dist30980 = dist.Normal(mu=-2, sigma=1)
		y30922 = 1 
		p30981 = dist30980.(y30922) if x30979 else 0 # from observe with if  
		x30982 = logical_trans( x30910 > -1)
		x30983 = logical_trans( x30910 > 0)
		x30984 = not logical_trans(x30983)
		x30985 = logical_trans( x30982 and x30984)
		dist30986 = dist.Normal(mu=-0.5, sigma=1)
		y30919 = 1 
		p30987 = dist30986.(y30919) if x30985 else 0 # from observe with if  
		x30988 = logical_trans( x30910 < 1)
		x30989 = logical_trans( x30910 > 0)
		x30990 = logical_trans( x30988 and x30989)
		dist30991 = dist.Normal(mu=0.5, sigma=1)
		y30913 = 1 
		p30992 = dist30991.(y30913) if x30990 else 0 # from observe with if  
		logp =  p30968 + p30974 + p30981 + p30987 + p30992  # total log joint 
		return logp # need to modify output format

