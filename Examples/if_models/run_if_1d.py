#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 15 Jan 2018, YZ

from pyfo.pyfoppl.foppl import imports
import Examples.if_models.if_1d as if_1d
from pyfo.inference.dhmc import DHMCSampler as dhmc

burn_in = 10 ** 2
n_samples = 10 ** 2
stepsize_range = [0.03,0.15]
n_step_range = [10, 20]

dhmc_ = dhmc(if_1d.model, 1)

stats = dhmc_.sample(n_samples=n_samples, burn_in=burn_in,stepsize_range=stepsize_range,n_step_range=n_step_range, print_stats=True)

# samples =  stats['samples']
# all_samples = stats['samples_wo_burin']
