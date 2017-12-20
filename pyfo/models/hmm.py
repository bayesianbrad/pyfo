import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{x32920 y32948 x32935}
	Arcs A:
	#{[x32935 y32948] [x32920 x32935]}
	Conditional densities P:
	x32920 -> (fn [] (categorical [0.3333333333333333 0.3333333333333333 0.3333333333333333]))
	x32935 -> (fn [x32920] (categorical (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.7 0.15 0.15]] x32920)))
	y32948 -> (fn [x32935] (nth (vector (normal -1.0 1.0) (normal 1.0 1.0) (normal 0.0 1.0)) x32935))
	Observed values O:
	y32948 -> 0.9
	'''

	@classmethod
	def gen_vars(self):
		return ['x32920', 'x32935']

	@classmethod
	def gen_cont_vars(self):
		return ['x32920', 'x32935']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x32977 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist32978 = dist.Categorical(ps=x32977)
		x32920 = dist32978.sample()   #sample 
		x32980 = [0.1,0.5,0.4]
		x32981 = [0.2,0.2,0.6]
		x32982 = [0.7,0.15,0.15]
		x32983 = [x32980,x32981,x32982]
		x32984 = x32983[int(x32920)]
		dist32985 = dist.Categorical(ps=x32984)
		x32935 = dist32985.sample()   #sample 
		x32987 = [int(x32935)]
		y32948 = 0.9 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x32977 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist32978 = dist.Categorical(ps=x32977)
		x32920 =  state['x32920']   # get the x from input arg
		p32979 = dist32978.log_pdf( x32920) # from prior
		x32980 = [0.1,0.5,0.4]
		x32981 = [0.2,0.2,0.6]
		x32982 = [0.7,0.15,0.15]
		x32983 = [x32980,x32981,x32982]
		x32984 = x32983[int(x32920)]
		dist32985 = dist.Categorical(ps=x32984)
		x32935 =  state['x32935']   # get the x from input arg
		p32986 = dist32985.log_pdf( x32935) # from prior
		x32987 = [int(x32935)]
		y32948 = 0.9 
		p32988 = x32987.log_pdf(y32948) # from observe  
		logp =  p32979 + p32986 + p32988  # total log joint 
		return logp # need to modify output format

