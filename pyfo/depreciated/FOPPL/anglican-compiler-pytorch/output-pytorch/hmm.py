import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y30853 x30821 x30838}
	Arcs A:
	#{[x30821 x30838] [x30838 y30853]}
	Conditional densities P:
	x30821 -> (fn [] (categorical [0.3333333333333333 0.3333333333333333 0.3333333333333333]))
	x30838 -> (fn [x30821] (categorical (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.7 0.15 0.15]] x30821)))
	y30853 -> (fn [x30838] (nth (vector (normal -1.0 1.0) (normal 1.0 1.0) (normal 0.0 1.0)) x30838))
	Observed values O:
	y30853 -> 0.9
	'''

	@classmethod
	def gen_vars(self):
		return ['x30821', 'x30838']

	@classmethod
	def gen_cont_vars(self):
		return ['x30821', 'x30838']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y30853', 'x30821', 'x30838']

	@classmethod
	def get_arcs(self):
		return [('x30821', 'x30838'), ('x30838', 'y30853')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x30886 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist30887 = dist.Categorical(ps=x30886)
		x30821 = dist30887.sample()   #sample 
		x30889 = [0.1,0.5,0.4]
		x30890 = [0.2,0.2,0.6]
		x30891 = [0.7,0.15,0.15]
		x30892 = [x30889,x30890,x30891]
		x30893 = x30892[int(x30821)]
		dist30894 = dist.Categorical(ps=x30893)
		x30838 = dist30894.sample()   #sample 
		dist30896 = dist.Normal(mu=-1.0, sigma=1.0)
		dist30897 = dist.Normal(mu=1.0, sigma=1.0)
		dist30898 = dist.Normal(mu=0.0, sigma=1.0)
		x30899 = [dist30896,dist30897,dist30898]
		x30900 = x30899[int(x30838)]
		y30853 = 0.9 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x30886 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist30887 = dist.Categorical(ps=x30886)
		x30821 =  state['x30821']   # get the x from input arg
		p30888 = dist30887.log_pdf( x30821) # from prior
		x30889 = [0.1,0.5,0.4]
		x30890 = [0.2,0.2,0.6]
		x30891 = [0.7,0.15,0.15]
		x30892 = [x30889,x30890,x30891]
		x30893 = x30892[int(x30821)]
		dist30894 = dist.Categorical(ps=x30893)
		x30838 =  state['x30838']   # get the x from input arg
		p30895 = dist30894.log_pdf( x30838) # from prior
		dist30896 = dist.Normal(mu=-1.0, sigma=1.0)
		dist30897 = dist.Normal(mu=1.0, sigma=1.0)
		dist30898 = dist.Normal(mu=0.0, sigma=1.0)
		x30899 = [dist30896,dist30897,dist30898]
		x30900 = x30899[int(x30838)]
		y30853 = 0.9 
		p30901 = x30900.log_pdf(y30853) # from observe  
		logp =  p30888 + p30895 + p30901  # total log joint 
		return logp # need to modify output format

