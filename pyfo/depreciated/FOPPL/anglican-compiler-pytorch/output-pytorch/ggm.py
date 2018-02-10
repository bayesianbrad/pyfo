import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y33266 x33261}
	Arcs A:
	#{[x33261 y33266]}
	Conditional densities P:
	x33261 -> (fn [] (categorical [0.3 0.7]))
	y33266 -> (fn [x33261] (normal (get [-5 5] x33261) 2))
	Observed values O:
	y33266 -> clojure.lang.LazySeq@137e8980
	'''

	@classmethod
	def gen_vars(self):
		return ['x33261']

	@classmethod
	def gen_cont_vars(self):
		return ['x33261']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y33266', 'x33261']

	@classmethod
	def get_arcs(self):
		return [('x33261', 'y33266')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x33293 = [0.3,0.7]
		dist33294 = dist.Categorical(ps=x33293)
		x33261 = dist33294.sample()   #sample 
		x33296 = [-5,5]
		x33297 = x33296[int(x33261)]
		dist33298 = dist.Normal(mu=x33297, sigma=2)
		x33299 = [-7,7]
		x33300 = x33299[int(x33261)]
		y33266 = x33300 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x33293 = [0.3,0.7]
		dist33294 = dist.Categorical(ps=x33293)
		x33261 =  state['x33261']   # get the x from input arg
		p33295 = dist33294.log_pdf( x33261) # from prior
		x33296 = [-5,5]
		x33297 = x33296[int(x33261)]
		dist33298 = dist.Normal(mu=x33297, sigma=2)
		x33299 = [-7,7]
		x33300 = x33299[int(x33261)]
		y33266 = x33300 
		p33301 = dist33298.log_pdf(y33266) # from observe  
		logp =  p33295 + p33301  # total log joint 
		return logp # need to modify output format

