import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):

	@staticmethod
	def gen_vars():
		return ['x49895'] # list

	@staticmethod
	def gen_cont_vars():
		return ['x49895'] # need to modify output format

	@staticmethod
	def gen_disc_vars():
		return ['[]'] # need to modify output format

	# prior samples 
	@staticmethod
	def gen_prior_samples(self):
		dist49921 = dist.Normal(mu=1.0, sigma=5.0)
		x49895 = dist49921.sample()   #sample 
		x49923 =  x49895 + 1  
		dist49924 = dist.Normal(mu=x49923, sigma=2.0)
		y49896 = 7.0 
		state = gen_vars.__func__() 
		state = locals()[state[0]]
		return state # list 
		
			# compute pdf 
	@staticmethod
	def gen_pdf(self, state):
		dist49921 = dist.Normal(mu=1.0, sigma=5.0)
		x49895 =  state['x49895']   # get the x from input arg
		p49922 = dist49921.logpdf( x49895) # from prior
		x49923 =  x49895 + 1  
		dist49924 = dist.Normal(mu=x49923, sigma=2.0)
		y49896 = 7.0 
		p49925 = dist49924.logpdf( y49896) # from observe  
		logp =  p49922 + p49925  # total log joint 
		return logp # need to modify output format
		
		