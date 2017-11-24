"""
A pseudo interface for now
"""
#PYTHON 3  type hints.
from pyfo.distributions import *
from typing import Dict, List, Bool
from torch.autograd import Variable
# prior samples
def gen_prior_samples() -> Dict[str, Variable]:
	dist31055 = Normal(mu=0, sigma=1)
	x31020 = dist31055.sample()   #sample
	x31057 = logical_trans( x31020 > 0)
	dist31058 = Normal(mean=1, std=1)
	y31023 = 1
	x31060 = logical_trans( x31020 > 0)
	x31061 = not logical_trans(x31060)
	dist31062 = Normal(mean=-1, std=1)
	y31026 = 1
	return Xs# need to modify output format

# compute pdf
def gen_pdf(Xs: Dict):
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
	return logp

def gen_cont_vars():
	return [x31020] # need to modify output format

def gen_disc_vars():
	return [[]] # need to modify output format

