def gen_vars():
return [x28076] # list

def gen_cont_vars():
return [x28076] # need to modify output format

def gen_disc_vars():
return [[]] # need to modify output format

# prior samples 
def gen_prior_samples():
dist28102 = Normal(mean=1.0, std=5.0)
x28076 = dist28102.sample()   #sample 
x28104 =  x28076 + 1  
dist28105 = Normal(mean=x28104, std=2.0)
y28077 = 7.0 
Xs = gen_vars() 
return Xs # list 

# compute pdf 
def gen_pdf(Xs, compute_grad = True):
dist28102 = Normal(mean=1.0, std=5.0)
x28076 =  Xs.get('x28076')   # get the x from input arg
p28103 = dist28102.logpdf( x28076) # from prior
x28104 =  x28076 + 1  
dist28105 = Normal(mean=x28104, std=2.0)
y28077 = 7.0 
p28106 = dist28105.logpdf( y28077) # from observe  
logp =  p28103 + p28106  # total log joint 
return logp # need to modify output format

