#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  12:40
Date created:  06/09/2017

License: MIT
'''
import importlib
from HMC.hmc_class import HMCsampler as HMC
from Utils.plotting_and_saving import Plotting
models = ['conjgauss', 'conditionalif', 'linearreg', 'hierarchial', 'mixture', 'condif2']

# for i in range(len(models)-2):
#     program     = getattr(importlib.import_module('Utils.program'), models[i])
#     # import Utils.program.model  == getattr(importlib.import_module("module.submodule"), "MyClass") #in theory a user would know
#     # which model they want to use and so this line would not be necessary.
#     hmcsampler  = HMC(program, burn_in=0, n_samples = 500)
#     samples, samples_with_burnin, mean =  hmcsampler.run_sampler()
#     plots = Plotting(samples,samples_with_burnin,mean, model = models[i])
#     plots.call_all_methods()

program = getattr(importlib.import_module('Utils.program'), models[5])
# import Utils.program.model  == getattr(importlib.import_module("module.submodule"), "MyClass") #in theory a user would know
# which model they want to use and so this line would not be necessary.
hmcsampler = HMC(program, burn_in=0, n_samples=5000)
samples, samples_with_burnin, mean = hmcsampler.run_sampler()
plots = Plotting(samples, samples_with_burnin, mean, model=models[5])
plots.call_all_methods()