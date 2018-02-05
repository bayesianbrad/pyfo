import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y33617 x33602 x33585}
	Arcs A:
	#{[x33585 x33602] [x33602 y33617]}
	Conditional densities P:
	x33585 -> (fn [] (categorical [0.3333333333333333 0.3333333333333333 0.3333333333333333]))
	x33602 -> (fn [x33585] (categorical (nth [[0.1 0.5 0.4] [0.2 0.2 0.6] [0.7 0.15 0.15]] x33585)))
	y33617 -> (fn [x33602] (nth (vector (normal -1.0 1.0) (normal 1.0 1.0) (normal 0.0 1.0)) x33602))
	Observed values O:
	y33617 -> 0.9
	'''

	@classmethod
	def gen_vars(self):
		return ['x33602', 'x33585']

	@classmethod
	def gen_cont_vars(self):
		return ['x33602', 'x33585']

	@classmethod
	def gen_disc_vars(self):
		return []

	@classmethod
	def get_vertices(self):
		return ['y33617', 'x33602', 'x33585']

	@classmethod
	def get_arcs(self):
		return [('x33585', 'x33602'), ('x33602', 'y33617')]

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		x33650 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist33651 = dist.Categorical(ps=x33650)
		x33585 = dist33651.sample()   #sample 
		x33653 = [0.1,0.5,0.4]
		x33654 = [0.2,0.2,0.6]
		x33655 = [0.7,0.15,0.15]
		x33656 = [x33653,x33654,x33655]
		x33657 = x33656[int(x33585)]
		dist33658 = dist.Categorical(ps=x33657)
		x33602 = dist33658.sample()   #sample 
		dist33660 = dist.Normal(mu=-1.0, sigma=1.0)
		dist33661 = dist.Normal(mu=1.0, sigma=1.0)
		dist33662 = dist.Normal(mu=0.0, sigma=1.0)
		x33663 = [dist33660,dist33661,dist33662]
		x33664 = x33663[int(x33602)]
		y33617 = 0.9 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		x33650 = [0.3333333333333333,0.3333333333333333,0.3333333333333333]
		dist33651 = dist.Categorical(ps=x33650)
		x33585 =  state['x33585']   # get the x from input arg
		p33652 = dist33651.log_pdf( x33585) # from prior
		x33653 = [0.1,0.5,0.4]
		x33654 = [0.2,0.2,0.6]
		x33655 = [0.7,0.15,0.15]
		x33656 = [x33653,x33654,x33655]
		x33657 = x33656[int(x33585)]
		dist33658 = dist.Categorical(ps=x33657)
		x33602 =  state['x33602']   # get the x from input arg
		p33659 = dist33658.log_pdf( x33602) # from prior
		dist33660 = dist.Normal(mu=-1.0, sigma=1.0)
		dist33661 = dist.Normal(mu=1.0, sigma=1.0)
		dist33662 = dist.Normal(mu=0.0, sigma=1.0)
		x33663 = [dist33660,dist33661,dist33662]
		x33664 = x33663[int(x33602)]
		y33617 = 0.9 
		p33665 = x33664.log_pdf(y33617) # from observe  
		logp =  p33652 + p33659 + p33665  # total log joint 
		return logp # need to modify output format

