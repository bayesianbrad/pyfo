import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface
from pyfo.utils.core import VariableCast
class model(interface):
	'''
	Vertices V:
	#{x28736 y28743 y28740 x28739}
	Arcs A:
	#{[x28739 y28740] [x28736 y28743] [x28736 y28740]}
	Conditional densities P:
	x28736 -> (fn [] (normal 0 1))
	x28739 -> (fn [] (normal 0 1))
	y28740 -> (fn [x28736 x28739] (if (> x28736 0) (normal x28739 1)))
	y28743 -> (fn [x28736] (if (not (> x28736 0)) (normal -1 1)))
	Observed values O:
	y28740 -> 1
	y28743 -> 1
	'''

	@classmethod
	def gen_vars(self):
		return ['x28736', 'x28739']

	@classmethod
	def gen_cont_vars(self):
		return []

	@classmethod
	def gen_disc_vars(self):
		return ['x28736','x28739']

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist28778 = dist.Normal(mu=0, sigma=1)
		x28739 = dist28778.sample()   #sample 
		dist28780 = dist.Normal(mu=0, sigma=1)
		x28736 = dist28780.sample()   #sample 
		x28782 = self.logical_trans( x28736 > 0)
		x28783 = not self.logical_trans(x28782)
		dist28784 = dist.Normal(mu=-1, sigma=1)
		y28743 = 1 
		x28786 = self.logical_trans( x28736 > 0)
		dist28787 = dist.Normal(mu=x28739, sigma=1)
		y28740 = 1 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist28778 = dist.Normal(mu=0, sigma=1)
		x28739 =  state['x28739']   # get the x from input arg
		p28779 = dist28778.log_pdf( x28739) # from prior
		dist28780 = dist.Normal(mu=0, sigma=1)
		x28736 =  state['x28736']   # get the x from input arg
		p28781 = dist28780.log_pdf( x28736) # from prior
		x28782 = self.logical_trans( x28736 > 0)
		x28783 = not self.logical_trans(x28782)
		dist28784 = dist.Normal(mu=-1, sigma=1)
		y28743 = 1 
		p28785 = dist28784.log_pdf(y28743) if x28783 else 0 # from observe with if
		x28786 = self.logical_trans( x28736 > 0)
		dist28787 = dist.Normal(mu=x28739, sigma=1)
		y28740 = 1 
		p28788 = dist28787.log_pdf(y28740) if x28786 else 0 # from observe with if  
		logp =  p28779 + p28781 + p28785 + p28788  # total log joint 
		return logp # need to modify output format
	
	@classmethod
	def logical_trans(self, var):
	   value = VariableCast(var)
	   if value.data[0]:
		   return True
	   else:
		   return False