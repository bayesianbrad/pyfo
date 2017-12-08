def gen_vars():
return [x29807,x29810] # list

def gen_cont_vars():
return [x29807,x29810] # need to modify output format

def gen_disc_vars():
return [[]] # need to modify output format

# prior samples 
def gen_prior_samples():
dist29918 = Normal(mean=0.0, std=10.0)
x29810 = dist29918.sample()   #sample 
dist29920 = Normal(mean=0.0, std=10.0)
x29807 = dist29920.sample()   #sample 
x29922 = x29807 * 1.0
x29923 =  x29922 + x29810  
dist29924 = Normal(mean=x29923, std=1.0)
y29817 = 2.1 
x29926 = x29807 * 2.0
x29927 =  x29926 + x29810  
dist29928 = Normal(mean=x29927, std=1.0)
y29838 = 3.9 
x29930 = x29807 * 3.0
x29931 =  x29930 + x29810  
dist29932 = Normal(mean=x29931, std=1.0)
y29863 = 5.3 
Xs = gen_vars() 
return Xs # list 

# compute pdf 
def gen_pdf(Xs, compute_grad = True):
dist29918 = Normal(mean=0.0, std=10.0)
x29810 =  Xs['x29810']   # get the x from input arg
p29919 = dist29918.logpdf( x29810) # from prior
dist29920 = Normal(mean=0.0, std=10.0)
x29807 =  Xs['x29807']   # get the x from input arg
p29921 = dist29920.logpdf( x29807) # from prior
x29922 = x29807 * 1.0
x29923 =  x29922 + x29810  
dist29924 = Normal(mean=x29923, std=1.0)
y29817 = 2.1 
p29925 = dist29924.logpdf( y29817) # from observe  
x29926 = x29807 * 2.0
x29927 =  x29926 + x29810  
dist29928 = Normal(mean=x29927, std=1.0)
y29838 = 3.9 
p29929 = dist29928.logpdf( y29838) # from observe  
x29930 = x29807 * 3.0
x29931 =  x29930 + x29810  
dist29932 = Normal(mean=x29931, std=1.0)
y29863 = 5.3 
p29933 = dist29932.logpdf( y29863) # from observe  
logp =  p29919 + p29921 + p29925 + p29929 + p29933  # total log joint 
return logp # need to modify output format

