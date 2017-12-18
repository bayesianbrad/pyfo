import torch 
import numpy as np  
from torch.autograd import Variable  
import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):

	@staticmethod
	def gen_vars():
		return ['x51639'] # list

	@staticmethod
	def gen_cont_vars():
		return ['x51639'] # need to modify output format

	@staticmethod
	def gen_disc_vars():
		return ['[]'] # need to modify output format

	# prior samples 
	@staticmethod
	def gen_prior_samples():
		dist51696 = dist.Normal(mu=0, sigma=1)
		x51639 = dist51696.sample()   #sample 
		x51698 = logical_trans( x51639 > -1)
		x51699 = not logical_trans(x51698)
		x51700 = logical_trans( x51639 > 0)
		x51701 = not logical_trans(x51700)
		x51702 = logical_trans( x51699 and x51701)
		dist51703 = dist.Normal(mu=-2, sigma=1)
		y51651 = 1 
		x51705 = logical_trans( x51639 < 1)
		x51706 = logical_trans( x51639 > 0)
		x51707 = logical_trans( x51705 and x51706)
		dist51708 = dist.Normal(mu=0.5, sigma=1)
		y51642 = 1 
		x51710 = logical_trans( x51639 > -1)
		x51711 = logical_trans( x51639 > 0)
		x51712 = not logical_trans(x51711)
		x51713 = logical_trans( x51710 and x51712)
		dist51714 = dist.Normal(mu=-0.5, sigma=1)
		y51648 = 1 
		x51716 = logical_trans( x51639 < 1)
		x51717 = not logical_trans(x51716)
		x51718 = logical_trans( x51639 > 0)
		x51719 = logical_trans( x51717 and x51718)
		dist51720 = dist.Normal(mu=2, sigma=1)
		y51645 = 1 
		state = gen_vars.__func__() 
		state = locals()[state[0]]
		return state # list 
		
			# compute pdf 
	@staticmethod
	def gen_pdf(state):
		dist51696 = dist.Normal(mu=0, sigma=1)
		x51639 =  state['x51639']   # get the x from input arg
		p51697 = dist51696.logpdf( x51639) # from prior
		x51698 = logical_trans( x51639 > -1)
		x51699 = not logical_trans(x51698)
		x51700 = logical_trans( x51639 > 0)
		x51701 = not logical_trans(x51700)
		x51702 = logical_trans( x51699 and x51701)
		dist51703 = dist.Normal(mu=-2, sigma=1)
		y51651 = 1 
		p51704 = dist51703.logpdf( y51651) if x51702 else 0 # from observe with if  
		x51705 = logical_trans( x51639 < 1)
		x51706 = logical_trans( x51639 > 0)
		x51707 = logical_trans( x51705 and x51706)
		dist51708 = dist.Normal(mu=0.5, sigma=1)
		y51642 = 1 
		p51709 = dist51708.logpdf( y51642) if x51707 else 0 # from observe with if  
		x51710 = logical_trans( x51639 > -1)
		x51711 = logical_trans( x51639 > 0)
		x51712 = not logical_trans(x51711)
		x51713 = logical_trans( x51710 and x51712)
		dist51714 = dist.Normal(mu=-0.5, sigma=1)
		y51648 = 1 
		p51715 = dist51714.logpdf( y51648) if x51713 else 0 # from observe with if  
		x51716 = logical_trans( x51639 < 1)
		x51717 = not logical_trans(x51716)
		x51718 = logical_trans( x51639 > 0)
		x51719 = logical_trans( x51717 and x51718)
		dist51720 = dist.Normal(mu=2, sigma=1)
		y51645 = 1 
		p51721 = dist51720.logpdf( y51645) if x51719 else 0 # from observe with if  
		logp =  p51697 + p51704 + p51709 + p51715 + p51721  # total log joint 
		return logp # need to modify output format
		
		