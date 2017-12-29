import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y30697 y30649 x30636 y30672 x30633}
	Arcs A:
	#{[x30633 y30697] [x30633 y30649] [x30636 y30672] [x30636 y30649] [x30636 y30697] [x30633 y30672]}
	Conditional densities P:
	x30633 -> (fn [] (normal 0.0 10.0))
	x30636 -> (fn [] (normal 0.0 10.0))
	y30649 -> (fn [x30633 x30636] (normal (+ (* x30633 1.0) x30636) 1.0))
	y30672 -> (fn [x30633 x30636] (normal (+ (* x30633 2.0) x30636) 1.0))
	y30697 -> (fn [x30633 x30636] (normal (+ (* x30633 3.0) x30636) 1.0))
	Observed values O:
	y30649 -> 2.1
	y30672 -> 3.9
	y30697 -> 5.3
	'''

	@classmethod
	def gen_vars(self):
		return ['x30636', 'x30633']

	@classmethod
	def gen_cont_vars(self):
		return ['x30636', 'x30633']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y30697', 'y30649', 'x30636', 'y30672', 'x30633']

	@classmethod
	def get_arcs(self):
		return [('x30633', 'y30697'), ('x30633', 'y30649'), ('x30636', 'y30672'), ('x30636', 'y30649'), ('x30636', 'y30697'), ('x30633', 'y30672')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30752 = dist.Normal(mu=0.0, sigma=10.0)
		x30633 = dist30752.sample()   #sample 
		dist30754 = dist.Normal(mu=0.0, sigma=10.0)
		x30636 = dist30754.sample()   #sample 
		x30756 = x30633 * 2.0
		x30757 =  x30756 + x30636  
		dist30758 = dist.Normal(mu=x30757, sigma=1.0)
		y30672 = 3.9 
		x30760 = x30633 * 1.0
		x30761 =  x30760 + x30636  
		dist30762 = dist.Normal(mu=x30761, sigma=1.0)
		y30649 = 2.1 
		x30764 = x30633 * 3.0
		x30765 =  x30764 + x30636  
		dist30766 = dist.Normal(mu=x30765, sigma=1.0)
		y30697 = 5.3 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30752 = dist.Normal(mu=0.0, sigma=10.0)
		x30633 =  state['x30633']   # get the x from input arg
		p30753 = dist30752.log_pdf( x30633) # from prior
		dist30754 = dist.Normal(mu=0.0, sigma=10.0)
		x30636 =  state['x30636']   # get the x from input arg
		p30755 = dist30754.log_pdf( x30636) # from prior
		x30756 = x30633 * 2.0
		x30757 =  x30756 + x30636  
		dist30758 = dist.Normal(mu=x30757, sigma=1.0)
		y30672 = 3.9 
		p30759 = dist30758.log_pdf(y30672) # from observe  
		x30760 = x30633 * 1.0
		x30761 =  x30760 + x30636  
		dist30762 = dist.Normal(mu=x30761, sigma=1.0)
		y30649 = 2.1 
		p30763 = dist30762.log_pdf(y30649) # from observe  
		x30764 = x30633 * 3.0
		x30765 =  x30764 + x30636  
		dist30766 = dist.Normal(mu=x30765, sigma=1.0)
		y30697 = 5.3 
		p30767 = dist30766.log_pdf(y30697) # from observe  
		logp =  p30753 + p30755 + p30759 + p30763 + p30767  # total log joint 
		return logp # need to modify output format

