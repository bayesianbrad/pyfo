def gen_vars():
return [x29767] # list

def gen_cont_vars():
return [x29767] # need to modify output format

def gen_disc_vars():
return [[]] # need to modify output format

# prior samples 
def gen_prior_samples():
dist29791 = Normal(mean=1.0, std=5.0)
x29767 = dist29791.sample()   #sample 
dist29793 = Normal(mean=x29767, std=2.0)
y29768 = 7.0 
Xs = gen_vars() 
return Xs # list 

# compute pdf 
def gen_pdf(Xs, compute_grad = True):
dist29791 = Normal(mean=1.0, std=5.0)
x29767 =  Xs['x29767']   # get the x from input arg
p29792 = dist29791.logpdf( x29767) # from prior
dist29793 = Normal(mean=x29767, std=2.0)
y29768 = 7.0 
p29794 = dist29793.logpdf( y29768) # from observe  
logp =  p29792 + p29794  # total log joint 
return logp # need to modify output format

