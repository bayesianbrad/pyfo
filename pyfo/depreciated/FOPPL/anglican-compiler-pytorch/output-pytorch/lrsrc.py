import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x33400 y33413 x33397 y33436 y33461}
	Arcs A:
	#{[x33400 y33436] [x33400 y33461] [x33397 y33413] [x33397 y33461] [x33397 y33436] [x33400 y33413]}
	Conditional densities P:
	x33397 -> (fn [] (normal 0.0 10.0))
	x33400 -> (fn [] (normal 0.0 10.0))
	y33413 -> (fn [x33400 x33397] (normal (+ (* x33397 1.0) x33400) 1.0))
	y33436 -> (fn [x33400 x33397] (normal (+ (* x33397 2.0) x33400) 1.0))
	y33461 -> (fn [x33400 x33397] (normal (+ (* x33397 3.0) x33400) 1.0))
	Observed values O:
	y33413 -> 2.1
	y33436 -> 3.9
	y33461 -> 5.3
	'''

	@classmethod
	def gen_vars(self):
		return ['x33400', 'x33397']

	@classmethod
	def gen_cont_vars(self):
		return ['x33400', 'x33397']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x33400', 'y33413', 'x33397', 'y33436', 'y33461']

	@classmethod
	def get_arcs(self):
		return [('x33400', 'y33436'), ('x33400', 'y33461'), ('x33397', 'y33413'), ('x33397', 'y33461'), ('x33397', 'y33436'), ('x33400', 'y33413')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist33516 = dist.Normal(mu=0.0, sigma=10.0)
		x33397 = dist33516.sample()   #sample 
		dist33518 = dist.Normal(mu=0.0, sigma=10.0)
		x33400 = dist33518.sample()   #sample 
		x33520 = x33397 * 1.0
		x33521 =  x33520 + x33400  
		dist33522 = dist.Normal(mu=x33521, sigma=1.0)
		y33413 = 2.1 
		x33524 = x33397 * 2.0
		x33525 =  x33524 + x33400  
		dist33526 = dist.Normal(mu=x33525, sigma=1.0)
		y33436 = 3.9 
		x33528 = x33397 * 3.0
		x33529 =  x33528 + x33400  
		dist33530 = dist.Normal(mu=x33529, sigma=1.0)
		y33461 = 5.3 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist33516 = dist.Normal(mu=0.0, sigma=10.0)
		x33397 =  state['x33397']   # get the x from input arg
		p33517 = dist33516.log_pdf( x33397) # from prior
		dist33518 = dist.Normal(mu=0.0, sigma=10.0)
		x33400 =  state['x33400']   # get the x from input arg
		p33519 = dist33518.log_pdf( x33400) # from prior
		x33520 = x33397 * 1.0
		x33521 =  x33520 + x33400  
		dist33522 = dist.Normal(mu=x33521, sigma=1.0)
		y33413 = 2.1 
		p33523 = dist33522.log_pdf(y33413) # from observe  
		x33524 = x33397 * 2.0
		x33525 =  x33524 + x33400  
		dist33526 = dist.Normal(mu=x33525, sigma=1.0)
		y33436 = 3.9 
		p33527 = dist33526.log_pdf(y33436) # from observe  
		x33528 = x33397 * 3.0
		x33529 =  x33528 + x33400  
		dist33530 = dist.Normal(mu=x33529, sigma=1.0)
		y33461 = 5.3 
		p33531 = dist33530.log_pdf(y33461) # from observe  
		logp =  p33517 + p33519 + p33523 + p33527 + p33531  # total log joint 
		return logp # need to modify output format

