import pyfo.distributions as dist
import pyfo.inference as interface

class model(interface):

    def __init__(self):
        super().__init__()

    def gen_vars(self):
        return ['x32870'] # list

    def gen_cont_vars(self):
        return ['x32870'] # need to modify output format

    def gen_disc_vars(self, disc_vars):
        return [[]] # need to modify output format

    # prior samples
    def gen_prior_samples(self):
        dist32896 = dist.Normal(mean=1.0, std=5.0)
        x32870 = dist32896.sample()   #sample
        x32898 =  x32870 + 1
        dist32899 = dist.Normal(mean=x32898, std=2.0)
        y32871 = 7.0

        return [x32870]

    # compute pdf
    def gen_pdf(self,x):
        # Notes: At this point, all x's need to be leaf nodes
        dist32896 = dist.Normal(mean=1.0, std=5.0)
        x32870 =  x['x32870']   # get the x from input arg
        p32897 = dist32896.logpdf( x32870) # from prior
        x32898 =  x32870 + 1
        dist32899 = dist.Normal(mean=x32898, std=2.0)
        y32871 = 7.0
        p32900 = dist32899.logpdf( y32871) # from observe
        logp =  p32897 + p32900  # total log joint
        return logp # need to modify output format

