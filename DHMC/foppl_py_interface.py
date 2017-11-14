"""
A pseudo interface for now
"""

def get_return_E(foppl-query)
	return E
	
def get_prior_samples(foppl-query)
 	return prior_samples
 	
def get_logp(xs)
	return logp
	
def get_grad(xcont)
	return grad_xcont

def get_cont_var(xs)
	"""
	return continuous variable list
	"""
	return x_cont
	
def get_disc_var(xs)
	"""
	return discrete variable list
	"""
	return x_disc
	
	
def get_discrete_var(xs)
	"""
	return variables from discrete distributions 
	"""
	return x_discrete

def get_var_map()
	"""
	map x gensym names with xs in dhmc
	"""
	return x_map