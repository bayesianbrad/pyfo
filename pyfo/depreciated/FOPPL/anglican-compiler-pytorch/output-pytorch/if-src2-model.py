import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y33170 x33164 y33176 y33167 y33173}
	Arcs A:
	#{[x33164 y33170] [x33164 y33173] [x33164 y33176] [x33164 y33167]}
	Conditional densities P:
	x33164 -> (fn [] (normal 0 1))
	y33167 -> (fn [x33164] (if (and (< x33164 1) (> x33164 0)) (normal 0.5 1)))
	y33170 -> (fn [x33164] (if (and (not (< x33164 1)) (> x33164 0)) (normal 2 1)))
	y33173 -> (fn [x33164] (if (and (> x33164 -1) (not (> x33164 0))) (normal -0.5 1)))
	y33176 -> (fn [x33164] (if (and (not (> x33164 -1)) (not (> x33164 0))) (normal -2 1)))
	Observed values O:
	y33167 -> 1
	y33170 -> 1
	y33173 -> 1
	y33176 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x33164']

	@classmethod
	def gen_cont_vars(self):
		return ['x33164']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y33170', 'x33164', 'y33176', 'y33167', 'y33173']

	@classmethod
	def get_arcs(self):
		return [('x33164', 'y33170'), ('x33164', 'y33173'), ('x33164', 'y33176'), ('x33164', 'y33167')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33221 = dist.Normal(mu=0, sigma=1)
		x33164 = dist33221.sample()   #sample 
		x33223 = logical_trans( x33164 > -1)
		x33224 = not logical_trans(x33223)
		x33225 = logical_trans( x33164 > 0)
		x33226 = not logical_trans(x33225)
		x33227 = logical_trans( x33224 and x33226)
		dist33228 = dist.Normal(mu=-2, sigma=1)
		y33176 = 1 
		x33230 = logical_trans( x33164 < 1)
		x33231 = logical_trans( x33164 > 0)
		x33232 = logical_trans( x33230 and x33231)
		dist33233 = dist.Normal(mu=0.5, sigma=1)
		y33167 = 1 
		x33235 = logical_trans( x33164 > -1)
		x33236 = logical_trans( x33164 > 0)
		x33237 = not logical_trans(x33236)
		x33238 = logical_trans( x33235 and x33237)
		dist33239 = dist.Normal(mu=-0.5, sigma=1)
		y33173 = 1 
		x33241 = logical_trans( x33164 < 1)
		x33242 = not logical_trans(x33241)
		x33243 = logical_trans( x33164 > 0)
		x33244 = logical_trans( x33242 and x33243)
		dist33245 = dist.Normal(mu=2, sigma=1)
		y33170 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33221 = dist.Normal(mu=0, sigma=1)
		x33164 =  state['x33164']   # get the x from input arg
		p33222 = dist33221.log_pdf( x33164) # from prior
		x33223 = logical_trans( x33164 > -1)
		x33224 = not logical_trans(x33223)
		x33225 = logical_trans( x33164 > 0)
		x33226 = not logical_trans(x33225)
		x33227 = logical_trans( x33224 and x33226)
		dist33228 = dist.Normal(mu=-2, sigma=1)
		y33176 = 1 
		p33229 = dist33228.(y33176) if x33227 else 0 # from observe with if  
		x33230 = logical_trans( x33164 < 1)
		x33231 = logical_trans( x33164 > 0)
		x33232 = logical_trans( x33230 and x33231)
		dist33233 = dist.Normal(mu=0.5, sigma=1)
		y33167 = 1 
		p33234 = dist33233.(y33167) if x33232 else 0 # from observe with if  
		x33235 = logical_trans( x33164 > -1)
		x33236 = logical_trans( x33164 > 0)
		x33237 = not logical_trans(x33236)
		x33238 = logical_trans( x33235 and x33237)
		dist33239 = dist.Normal(mu=-0.5, sigma=1)
		y33173 = 1 
		p33240 = dist33239.(y33173) if x33238 else 0 # from observe with if  
		x33241 = logical_trans( x33164 < 1)
		x33242 = not logical_trans(x33241)
		x33243 = logical_trans( x33164 > 0)
		x33244 = logical_trans( x33242 and x33243)
		dist33245 = dist.Normal(mu=2, sigma=1)
		y33170 = 1 
		p33246 = dist33245.(y33170) if x33244 else 0 # from observe with if  
		logp =  p33222 + p33229 + p33234 + p33240 + p33246  # total log joint 
		return logp # need to modify output format

