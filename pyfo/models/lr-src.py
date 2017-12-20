import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
from pyfo.utils.interface import interface

class model(interface):
	'''
	Vertices V:
	#{y32796 y32750 x32743 y32771 x32740}
	Arcs A:
	#{[x32743 y32771] [x32743 y32796] [x32743 y32750] [x32740 y32750] [x32740 y32796] [x32740 y32771]}
	Conditional densities P:
	x32740 -> (fn [] (normal 0.0 10.0))
	x32743 -> (fn [] (normal 0.0 10.0))
	y32750 -> (fn [x32740 x32743] (normal (+ (* x32740 1.0) x32743) 1.0))
	y32771 -> (fn [x32740 x32743] (normal (+ (* x32740 2.0) x32743) 1.0))
	y32796 -> (fn [x32740 x32743] (normal (+ (* x32740 3.0) x32743) 1.0))
	Observed values O:
	y32750 -> 2.1
	y32771 -> 3.9
	y32796 -> 5.3
	'''

	@classmethod
	def gen_vars(self):
		return ['x32743', 'x32740']

	@classmethod
	def gen_cont_vars(self):
		return ['x32743', 'x32740']

	@classmethod
	def gen_disc_vars(self):
		return []

	# prior samples 
	@classmethod
	def gen_prior_samples(self):
		dist32851 = dist.Normal(mu=0.0, sigma=10.0)
		x32740 = dist32851.sample()   #sample 
		dist32853 = dist.Normal(mu=0.0, sigma=10.0)
		x32743 = dist32853.sample()   #sample 
		x32855 = x32740 * 2.0
		x32856 =  x32855 + x32743  
		dist32857 = dist.Normal(mu=x32856, sigma=1.0)
		y32771 = 3.9 
		x32859 = x32740 * 1.0
		x32860 =  x32859 + x32743  
		dist32861 = dist.Normal(mu=x32860, sigma=1.0)
		y32750 = 2.1 
		x32863 = x32740 * 3.0
		x32864 =  x32863 + x32743  
		dist32865 = dist.Normal(mu=x32864, sigma=1.0)
		y32796 = 5.3 
		state = {}
		for _gv in self.gen_vars():
			state[_gv] = locals()[_gv]
		return state # dictionary

	# compute pdf 
	@classmethod
	def gen_pdf(self, state):
		dist32851 = dist.Normal(mu=0.0, sigma=10.0)
		x32740 =  state['x32740']   # get the x from input arg
		p32852 = dist32851.log_pdf( x32740) # from prior
		dist32853 = dist.Normal(mu=0.0, sigma=10.0)
		x32743 =  state['x32743']   # get the x from input arg
		p32854 = dist32853.log_pdf( x32743) # from prior
		x32855 = x32740 * 2.0
		x32856 =  x32855 + x32743  
		dist32857 = dist.Normal(mu=x32856, sigma=1.0)
		y32771 = 3.9 
		p32858 = dist32857.log_pdf(y32771) # from observe  
		x32859 = x32740 * 1.0
		x32860 =  x32859 + x32743  
		dist32861 = dist.Normal(mu=x32860, sigma=1.0)
		y32750 = 2.1 
		p32862 = dist32861.log_pdf(y32750) # from observe  
		x32863 = x32740 * 3.0
		x32864 =  x32863 + x32743  
		dist32865 = dist.Normal(mu=x32864, sigma=1.0)
		y32796 = 5.3 
		p32866 = dist32865.log_pdf(y32796) # from observe  
		logp =  p32852 + p32854 + p32858 + p32862 + p32866  # total log joint 
		return logp # need to modify output format

