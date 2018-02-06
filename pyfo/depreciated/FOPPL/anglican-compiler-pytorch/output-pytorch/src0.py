import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x30125 y30126}
	Arcs A:
	#{[x30125 y30126]}
	Conditional densities P:
	x30125 -> (fn [] (normal 1.0 5.0))
	y30126 -> (fn [x30125] (normal x30125 2.0))
	Observed values O:
	y30126 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x30125']

	@classmethod
	def gen_cont_vars(self):
		return ['x30125']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x30125', 'y30126']

	@classmethod
	def get_arcs(self):
		return [('x30125', 'y30126')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30149 = dist.Normal(mu=1.0, sigma=5.0)
		x30125 = dist30149.sample()   #sample 
		dist30151 = dist.Normal(mu=x30125, sigma=2.0)
		y30126 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30149 = dist.Normal(mu=1.0, sigma=5.0)
		x30125 =  state['x30125']   # get the x from input arg
		p30150 = dist30149.log_pdf( x30125) # from prior
		dist30151 = dist.Normal(mu=x30125, sigma=2.0)
		y30126 = 7.0 
		p30152 = dist30151.log_pdf(y30126) # from observe  
		logp =  p30150 + p30152  # total log joint 
		return logp # need to modify output format

