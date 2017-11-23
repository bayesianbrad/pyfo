"""
A pseudo interface for now
"""
#PYTHON 3  type hints.
from pyfo.distributions import *

# prior samples
def gen_prior_samples():
	dist31055 = Normal(mu=0, sigma=1)
	x31020 = dist31055.sample()   #sample
	x31057 = logical_trans( x31020 > 0)
	dist31058 = Normal(mean=1, std=1)
	y31023 = 1
	x31060 = logical_trans( x31020 > 0)
	x31061 = not logical_trans(x31060)
	dist31062 = Normal(mean=-1, std=1)
	y31026 = 1
	Xs = gen_ordered_vars()

	# if Xs is a list of strings, then using the following makes a list of dict.fromkeys(Xs) {'l1':None, 'l2':None ....} Etc

	dic = dict.fromkeys(Xs)
	for k1 in state.keys():
		for k2 in state[k1]:
			print(' This is the key {0} for the main state and this is the subkey {1} with value {2}'.format(k1, k2,
																											 state[k1][
																												 k2]))

	#Ideally we would have First hashmap : {'params': {},'obs': {}}
	# then populate it with

	# state  = {'params': {'x1':<val>, 'x2':<val> , ...},'obs':{y31023:<val>}}
	#  ''

	return Xs # need to modify output format

# compute pdf
def gen_pdf(Xs, compute_grad = True):
	dist31055 = Normal(mean=0, std=1)
	x31020 =  Xs[var_x_map.get('x31020')]   # get the x from input arg
	p31056 = dist31055.logpdf( x31020) # from prior
	x31057 = logical_trans( x31020 > 0)
	dist31058 = Normal(mean=1, std=1)
	y31023 = 1
	p31059 = dist31058.logpdf( y31023) if x31057 else 0 # from observe with if
	x31060 = logical_trans( x31020 > 0)
	x31061 = not logical_trans(x31060)
	dist31062 = Normal(mean=-1, std=1)
	y31026 = 1
	p31063 = dist31062.logpdf( y31026) if x31061 else 0 # from observe with if
	logp =  p31056 + p31059 + p31063  # total log joint
	var_cont = gen_cont_vars()
	if compute_grad:
		grad = torch.autograd.grad(logp, var_cont)[0] # need to modify format
	return logp, grad # need to modify output format

def gen_cont_vars():
return [x31020] # need to modify output format

def gen_disc_vars():
return [[]] # need to modify output format

