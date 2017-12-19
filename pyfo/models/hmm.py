import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{x30830 y30858 x30845}
	Arcs A:
	#{[x30830 x30845] [x30845 y30858]}
	Conditional densities P:
	x30830 -> (fn [] (categorical [0.3333333333333333 0.3333333333333333 0.3333333333333333]))
	x30845 -> (fn [x30830] (categorical (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.7 0.15 0.15]] x30830)))
	y30858 -> (fn [x30845] (nth (vector (normal -1.0 1.0) (normal 1.0 1.0) (normal 0.0 1.0)) x30845))
	Observed values O:
	y30858 -> 0.9
	'''

	@classmethod
	def gen_vars(self):
		return ['x30830', 'x30845']

	@classmethod
	def gen_cont_vars(self):
		return ['x30830', 'x30845']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x30887 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist30888 = dist.Categorical(p=x30887)
		x30830 = dist30888.sample()   #sample 
		x30890 = [0.1,0.5,0.4]
		x30891 = [0.2,0.2,0.6]
		x30892 = [0.7,0.15,0.15]
		x30893 = [x30890,x30891,x30892]
		x30894 = x30893[int(x30830)]
		dist30895 = dist.Categorical(p=x30894)
		x30845 = dist30895.sample()   #sample 
		x30897 = [int(x30845)]
		y30858 = 0.9 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x30887 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist30888 = dist.Categorical(p=x30887)
		x30830 =  state['x30830']   # get the x from input arg
		p30889 = dist30888.logpdf( x30830) # from prior
		x30890 = [0.1,0.5,0.4]
		x30891 = [0.2,0.2,0.6]
		x30892 = [0.7,0.15,0.15]
		x30893 = [x30890,x30891,x30892]
		x30894 = x30893[int(x30830)]
		dist30895 = dist.Categorical(p=x30894)
		x30845 =  state['x30845']   # get the x from input arg
		p30896 = dist30895.logpdf( x30845) # from prior
		x30897 = [int(x30845)]
		y30858 = 0.9 
		p30898 = x30897.logpdf(y30858) # from observe  
		logp =  p30889 + p30896 + p30898  # total log joint 
		return logp # need to modify output format

