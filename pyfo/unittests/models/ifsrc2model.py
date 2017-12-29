import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y32514 y32523 x32511 y32520 y32517}
	Arcs A:
	#{[x32511 y32523] [x32511 y32520] [x32511 y32514] [x32511 y32517]}
	Conditional densities P:
	x32511 -> (fn [] (normal 0 1))
	y32514 -> (fn [x32511] (if (and (< x32511 1) (> x32511 0)) (normal 0.5 1)))
	y32517 -> (fn [x32511] (if (and (not (< x32511 1)) (> x32511 0)) (normal 2 1)))
	y32520 -> (fn [x32511] (if (and (> x32511 -1) (not (> x32511 0))) (normal -0.5 1)))
	y32523 -> (fn [x32511] (if (and (not (> x32511 -1)) (not (> x32511 0))) (normal -2 1)))
	Observed values O:
	y32514 -> 1
	y32517 -> 1
	y32520 -> 1
	y32523 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x32511']

	@classmethod
	def gen_cont_vars(self):
		return ['x32511']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist32568 = dist.Normal(mu=0, sigma=1)
		x32511 = dist32568.sample()   #sample 
		x32570 = logical_trans( x32511 > -1)
		x32571 = logical_trans( x32511 > 0)
		x32572 = not logical_trans(x32571)
		x32573 = logical_trans( x32570 and x32572)
		dist32574 = dist.Normal(mu=-0.5, sigma=1)
		y32520 = 1 
		x32576 = logical_trans( x32511 < 1)
		x32577 = not logical_trans(x32576)
		x32578 = logical_trans( x32511 > 0)
		x32579 = logical_trans( x32577 and x32578)
		dist32580 = dist.Normal(mu=2, sigma=1)
		y32517 = 1 
		x32582 = logical_trans( x32511 > -1)
		x32583 = not logical_trans(x32582)
		x32584 = logical_trans( x32511 > 0)
		x32585 = not logical_trans(x32584)
		x32586 = logical_trans( x32583 and x32585)
		dist32587 = dist.Normal(mu=-2, sigma=1)
		y32523 = 1 
		x32589 = logical_trans( x32511 < 1)
		x32590 = logical_trans( x32511 > 0)
		x32591 = logical_trans( x32589 and x32590)
		dist32592 = dist.Normal(mu=0.5, sigma=1)
		y32514 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist32568 = dist.Normal(mu=0, sigma=1)
		x32511 =  state['x32511']   # get the x from input arg
		p32569 = dist32568.log_pdf( x32511) # from prior
		x32570 = logical_trans( x32511 > -1)
		x32571 = logical_trans( x32511 > 0)
		x32572 = not logical_trans(x32571)
		x32573 = logical_trans( x32570 and x32572)
		dist32574 = dist.Normal(mu=-0.5, sigma=1)
		y32520 = 1 
		p32575 = dist32574.(y32520) if x32573 else 0 # from observe with if  
		x32576 = logical_trans( x32511 < 1)
		x32577 = not logical_trans(x32576)
		x32578 = logical_trans( x32511 > 0)
		x32579 = logical_trans( x32577 and x32578)
		dist32580 = dist.Normal(mu=2, sigma=1)
		y32517 = 1 
		p32581 = dist32580.(y32517) if x32579 else 0 # from observe with if  
		x32582 = logical_trans( x32511 > -1)
		x32583 = not logical_trans(x32582)
		x32584 = logical_trans( x32511 > 0)
		x32585 = not logical_trans(x32584)
		x32586 = logical_trans( x32583 and x32585)
		dist32587 = dist.Normal(mu=-2, sigma=1)
		y32523 = 1 
		p32588 = dist32587.(y32523) if x32586 else 0 # from observe with if  
		x32589 = logical_trans( x32511 < 1)
		x32590 = logical_trans( x32511 > 0)
		x32591 = logical_trans( x32589 and x32590)
		dist32592 = dist.Normal(mu=0.5, sigma=1)
		y32514 = 1 
		p32593 = dist32592.(y32514) if x32591 else 0 # from observe with if  
		logp =  p32569 + p32575 + p32581 + p32588 + p32593  # total log joint 
		return logp # need to modify output format

