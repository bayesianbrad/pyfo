import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):
	'''
	Vertices V:
	#{y21419 x21407 y21416 y21413 y21410}
	Arcs A:
	#{[x21407 y21416] [x21407 y21419] [x21407 y21413] [x21407 y21410]}
	Conditional densities P:
	x21407 -> (fn [] (normal 0 1))
	y21410 -> (fn [x21407] (if (and (< x21407 1) (> x21407 0)) (normal 0.5 1)))
	y21413 -> (fn [x21407] (if (and (not (< x21407 1)) (> x21407 0)) (normal 2 1)))
	y21416 -> (fn [x21407] (if (and (> x21407 -1) (not (> x21407 0))) (normal -0.5 1)))
	y21419 -> (fn [x21407] (if (and (not (> x21407 -1)) (not (> x21407 0))) (normal -2 1)))
	Observed values O:
	y21410 -> 1
	y21413 -> 1
	y21416 -> 1
	y21419 -> 1
	'''

	def gen_vars(self):
		return ['x21407']

	def gen_cont_vars(self):
		return ['x21407']

	def gen_disc_vars(self):
		return []

	# prior samples 
	def gen_prior_samples(self):
		dist21464 = dist.Normal(mu=0, sigma=1)
		x21407 = dist21464.sample()   #sample 
		x21466 = logical_trans( x21407 > -1)
		x21467 = logical_trans( x21407 > 0)
		x21468 = not logical_trans(x21467)
		x21469 = logical_trans( x21466 and x21468)
		dist21470 = dist.Normal(mu=-0.5, sigma=1)
		y21416 = 1 
		x21472 = logical_trans( x21407 < 1)
		x21473 = not logical_trans(x21472)
		x21474 = logical_trans( x21407 > 0)
		x21475 = logical_trans( x21473 and x21474)
		dist21476 = dist.Normal(mu=2, sigma=1)
		y21413 = 1 
		x21478 = logical_trans( x21407 < 1)
		x21479 = logical_trans( x21407 > 0)
		x21480 = logical_trans( x21478 and x21479)
		dist21481 = dist.Normal(mu=0.5, sigma=1)
		y21410 = 1 
		x21483 = logical_trans( x21407 > -1)
		x21484 = not logical_trans(x21483)
		x21485 = logical_trans( x21407 > 0)
		x21486 = not logical_trans(x21485)
		x21487 = logical_trans( x21484 and x21486)
		dist21488 = dist.Normal(mu=-2, sigma=1)
		y21419 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	def gen_pdf(self, state):
		dist21464 = dist.Normal(mu=0, sigma=1)
		x21407 =  state['x21407']   # get the x from input arg
		p21465 = dist21464.logpdf( x21407) # from prior
		x21466 = logical_trans( x21407 > -1)
		x21467 = logical_trans( x21407 > 0)
		x21468 = not logical_trans(x21467)
		x21469 = logical_trans( x21466 and x21468)
		dist21470 = dist.Normal(mu=-0.5, sigma=1)
		y21416 = 1 
		p21471 = dist21470.logpdf(y21416) if x21469 else 0 # from observe with if  
		x21472 = logical_trans( x21407 < 1)
		x21473 = not logical_trans(x21472)
		x21474 = logical_trans( x21407 > 0)
		x21475 = logical_trans( x21473 and x21474)
		dist21476 = dist.Normal(mu=2, sigma=1)
		y21413 = 1 
		p21477 = dist21476.logpdf(y21413) if x21475 else 0 # from observe with if  
		x21478 = logical_trans( x21407 < 1)
		x21479 = logical_trans( x21407 > 0)
		x21480 = logical_trans( x21478 and x21479)
		dist21481 = dist.Normal(mu=0.5, sigma=1)
		y21410 = 1 
		p21482 = dist21481.logpdf(y21410) if x21480 else 0 # from observe with if  
		x21483 = logical_trans( x21407 > -1)
		x21484 = not logical_trans(x21483)
		x21485 = logical_trans( x21407 > 0)
		x21486 = not logical_trans(x21485)
		x21487 = logical_trans( x21484 and x21486)
		dist21488 = dist.Normal(mu=-2, sigma=1)
		y21419 = 1 
		p21489 = dist21488.logpdf(y21419) if x21487 else 0 # from observe with if  
		logp =  p21465 + p21471 + p21477 + p21482 + p21489  # total log joint 
		return logp # need to modify output format

