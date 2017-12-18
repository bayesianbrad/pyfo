import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):

	@staticmethod
	def gen_vars():
		return ['x51554'] # list

	@staticmethod
	def gen_cont_vars():
		return ['x51554'] # need to modify output format

	@staticmethod
	def gen_disc_vars():
		return ['[]'] # need to modify output format

	# prior samples 
	@staticmethod
	def gen_prior_samples():
		dist51578 = dist.Normal(mu=1.0, sigma=5.0)
		x51554 = dist51578.sample()   #sample 
		x51580 =  x51554 + 1  
		dist51581 = dist.Normal(mu=x51580, sigma=2.0)
		y51555 = 7.0 
		state = gen_vars.__func__() 
		state = locals()[state[0]]
		return state # list 
		
			# compute pdf 
	@staticmethod
	def gen_pdf(state):
		dist51578 = dist.Normal(mu=1.0, sigma=5.0)
		x51554 =  state['x51554']   # get the x from input arg
		p51579 = dist51578.logpdf( x51554) # from prior
		x51580 =  x51554 + 1  
		dist51581 = dist.Normal(mu=x51580, sigma=2.0)
		y51555 = 7.0 
		p51582 = dist51581.logpdf( y51555) # from observe  
		logp =  p51579 + p51582  # total log joint 
		return logp # need to modify output format
		
		