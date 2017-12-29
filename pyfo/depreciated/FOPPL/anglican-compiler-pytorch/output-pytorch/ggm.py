import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x30497 y30502}
	Arcs A:
	#{[x30497 y30502]}
	Conditional densities P:
	x30497 -> (fn [] (categorical [0.3 0.7]))
	y30502 -> (fn [x30497] (normal (get [-5 5] x30497) 2))
	Observed values O:
	y30502 -> clojure.lang.LazySeq@147ad982
	'''

	@classmethod
	def gen_vars(self):
		return ['x30497']

	@classmethod
	def gen_cont_vars(self):
		return ['x30497']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['x30497', 'y30502']

	@classmethod
	def get_arcs(self):
		return [('x30497', 'y30502')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x30529 = [0.3,0.7]
		dist30530 = dist.Categorical(ps=x30529)
		x30497 = dist30530.sample()   #sample 
		x30532 = [-5,5]
		x30533 = x30532[int(x30497)]
		dist30534 = dist.Normal(mu=x30533, sigma=2)
		x30535 = [-7,7]
		x30536 = x30535[int(x30497)]
		y30502 = x30536 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x30529 = [0.3,0.7]
		dist30530 = dist.Categorical(ps=x30529)
		x30497 =  state['x30497']   # get the x from input arg
		p30531 = dist30530.log_pdf( x30497) # from prior
		x30532 = [-5,5]
		x30533 = x30532[int(x30497)]
		dist30534 = dist.Normal(mu=x30533, sigma=2)
		x30535 = [-7,7]
		x30536 = x30535[int(x30497)]
		y30502 = x30536 
		p30537 = dist30534.log_pdf(y30502) # from observe  
		logp =  p30531 + p30537  # total log joint 
		return logp # need to modify output format

