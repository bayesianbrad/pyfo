def gen_vars():
return [x32870] # list

def gen_cont_vars():
return [x32870] # need to modify output format

def gen_disc_vars():
return [[]] # need to modify output format

# prior samples 
def gen_prior_samples():
dist32896 = Normal(mean=1.0, std=5.0)
x32870 = dist32896.sample()   #sample 
x32898 =  x32870 + 1  
dist32899 = Normal(mean=x32898, std=2.0)
y32871 = 7.0 
Xs = gen_vars() 
return Xs # list 

# compute pdf 
def gen_pdf(Xs, compute_grad = True):
dist32896 = Normal(mean=1.0, std=5.0)
x32870 =  Xs['x32870']   # get the x from input arg
p32897 = dist32896.logpdf( x32870) # from prior
x32898 =  x32870 + 1  
dist32899 = Normal(mean=x32898, std=2.0)
y32871 = 7.0 
p32900 = dist32899.logpdf( y32871) # from observe  
logp =  p32897 + p32900  # total log joint 
return logp # need to modify output format

