import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x30776 y30777}
	Arcs A:
	#{[x30776 y30777]}
	Conditional densities P:
	x30776 -> (fn [] (normal 1.0 5.0))
	y30777 -> (fn [x30776] (normal x30776 2.0))
	Observed values O:
	y30777 -> 7.0
	'''

	@classmethod
	def gen_vars(self):
		return ['x30776']

	@classmethod
	def gen_cont_vars(self):
		return ['x30776']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x30776', 'y30777']

	@classmethod
	def get_arcs(self):
		return [('x30776', 'y30777')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist30800 = dist.Normal(mu=1.0, sigma=5.0)
		x30776 = dist30800.sample()   #sample 
		dist30802 = dist.Normal(mu=x30776, sigma=2.0)
		y30777 = 7.0 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist30800 = dist.Normal(mu=1.0, sigma=5.0)
		x30776 =  state['x30776']   # get the x from input arg
		p30801 = dist30800.log_pdf( x30776) # from prior
		dist30802 = dist.Normal(mu=x30776, sigma=2.0)
		y30777 = 7.0 
		p30803 = dist30802.log_pdf(y30777) # from observe  
		logp =  p30801 + p30803  # total log joint 
		return logp # need to modify output format

