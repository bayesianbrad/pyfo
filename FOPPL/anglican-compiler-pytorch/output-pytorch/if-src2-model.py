def gen_vars():
return [x32957] # list

def gen_cont_vars():
return [x32957] # need to modify output format

def gen_disc_vars():
return [[]] # need to modify output format

# prior samples 
def gen_prior_samples():
dist33014 = Normal(mean=0, std=1)
x32957 = dist33014.sample()   #sample 
x33016 = logical_trans( x32957 > -1)
x33017 = logical_trans( x32957 > 0)
x33018 = not logical_trans(x33017)
x33019 = logical_trans( x33016 and x33018)
dist33020 = Normal(mean=-0.5, std=1)
y32966 = 1 
x33022 = logical_trans( x32957 < 1)
x33023 = logical_trans( x32957 > 0)
x33024 = logical_trans( x33022 and x33023)
dist33025 = Normal(mean=0.5, std=1)
y32960 = 1 
x33027 = logical_trans( x32957 > -1)
x33028 = not logical_trans(x33027)
x33029 = logical_trans( x32957 > 0)
x33030 = not logical_trans(x33029)
x33031 = logical_trans( x33028 and x33030)
dist33032 = Normal(mean=-2, std=1)
y32969 = 1 
x33034 = logical_trans( x32957 < 1)
x33035 = not logical_trans(x33034)
x33036 = logical_trans( x32957 > 0)
x33037 = logical_trans( x33035 and x33036)
dist33038 = Normal(mean=2, std=1)
y32963 = 1 
Xs = gen_vars() 
return Xs # list 

# compute pdf 
def gen_pdf(Xs, compute_grad = True):
dist33014 = Normal(mean=0, std=1)
x32957 =  Xs['x32957']   # get the x from input arg
p33015 = dist33014.logpdf( x32957) # from prior
x33016 = logical_trans( x32957 > -1)
x33017 = logical_trans( x32957 > 0)
x33018 = not logical_trans(x33017)
x33019 = logical_trans( x33016 and x33018)
dist33020 = Normal(mean=-0.5, std=1)
y32966 = 1 
p33021 = dist33020.logpdf( y32966) if x33019 else 0 # from observe with if  
x33022 = logical_trans( x32957 < 1)
x33023 = logical_trans( x32957 > 0)
x33024 = logical_trans( x33022 and x33023)
dist33025 = Normal(mean=0.5, std=1)
y32960 = 1 
p33026 = dist33025.logpdf( y32960) if x33024 else 0 # from observe with if  
x33027 = logical_trans( x32957 > -1)
x33028 = not logical_trans(x33027)
x33029 = logical_trans( x32957 > 0)
x33030 = not logical_trans(x33029)
x33031 = logical_trans( x33028 and x33030)
dist33032 = Normal(mean=-2, std=1)
y32969 = 1 
p33033 = dist33032.logpdf( y32969) if x33031 else 0 # from observe with if  
x33034 = logical_trans( x32957 < 1)
x33035 = not logical_trans(x33034)
x33036 = logical_trans( x32957 > 0)
x33037 = logical_trans( x33035 and x33036)
dist33038 = Normal(mean=2, std=1)
y32963 = 1 
p33039 = dist33038.logpdf( y32963) if x33037 else 0 # from observe with if  
logp =  p33015 + p33021 + p33026 + p33033 + p33039  # total log joint 
return logp # need to modify output format

