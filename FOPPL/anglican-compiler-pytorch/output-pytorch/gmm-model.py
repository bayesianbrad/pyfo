import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):

	@staticmethod
	def gen_vars():
		return ['x51734'] # list

	@staticmethod
	def gen_cont_vars():
		return ['x51734'] # need to modify output format

	@staticmethod
	def gen_disc_vars():
		return ['[]'] # need to modify output format

	# prior samples 
	@staticmethod
	def gen_prior_samples():
		dist51768 = dist.Normal(mu=, sigma=2)
		y51741 =  
		x51770 = [0.3,0.7]
		dist51771 = dist.Categorical(p=x51770)
		x51734 = dist51771.sample()   #sample 
		state = gen_vars.__func__() 
		state = locals()[state[0]]
		return state # list 
		
			# compute pdf 
	@staticmethod
	def gen_pdf(state):
		dist51768 = dist.Normal(mu=, sigma=2)
		y51741 =  
		p51769 = dist51768.logpdf( y51741) # from observe  
		x51770 = [0.3,0.7]
		dist51771 = dist.Categorical(p=x51770)
		x51734 =  state['x51734']   # get the x from input arg
		p51772 = dist51771.logpdf( x51734) # from prior
		logp =  p51769 + p51772  # total log joint 
		return logp # need to modify output format
		
		