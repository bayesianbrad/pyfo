#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  16:31
Date created:  19/12/2017

License: MIT


Notes for self:


The importer protocol
---------------------

The __import__ funtion takes 4 args.
1 -  name of module being imported, can be dotted
2 - A reference to the current global namespace

The protocol involves two objects a finder and a loader.

the finder object has a single method:
        finder.find_module(fulllname, path=None)
                - if finder is installed on the sys.meta_path, it will recieve a second argument
                which is None for a top-lvel module or package.__path__ for submodules or subpackages


the loader object also has one method:
        loader.load_module(fullname)

The loader method has a few responsibilities to filL:
        - __file__ attribute must be a string and set
        -__name__ atribute must be set
        -If it is a package the __path__ variable must be set.
        - __loader__ attribute must be set to the loader object
        - __package__ attribute must be set

        example:
        import importlib
        # Consider using importlib.util.module_for_loader() to handle
        # most of these details for you.
        def load_module(self, fullname):
            code = self.get_code(fullname)
            ispkg = self.is_package(fullname)
            mod = sys.modules.setdefault(fullname, imp.new_module(fullname))
            mod.__file__ = "<%s>" % self.__class__.__name__
            mod.__loader__ = self
            if ispkg:
                mod.__path__ = []
                mod.__package__ = fullname
            else:
                mod.__package__ = fullname.rpartition('.')[0]
            exec(code, mod.__dict__)
            return mod

'''

from pyfo.unittests.models.embedding.embedding_model import model
from pyfo.inference.dhmc import DHMCSampler as dhmc
import pyfo.distributions as dist
import numpy as np

# dhmc_ = dhmc(model)
# burn_in = 1
# n_sample = 10 ** 1
# stepsize_range = [0.03,0.15]
# n_step_range = [10, 20]
#
# stats = dhmc_.sample(n_samples=n_sample,burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range)
# samples = stats['samples']
# means = stats['means']
# print(means)
# print(stats['accept_prob'])


# print(dist.Bernoulli(0.7).sample())
obj1 = dist.Categorical(ps=[0.1, 0.2, 0.7], vs=[1,2,3])
# print(obj1.sample())
print(dist.Categorical(ps=[0.1, 0.2, 0.7], vs=[2,3,4]).sample())