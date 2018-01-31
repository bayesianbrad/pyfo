#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  11:07
Date created:  20/01/2018

License: MIT
'''

from pyfo.pyfoppl.foppl import imports
import pyfo.unittests.models.hmm.hmm as test
from pyfo.inference.dhmc import DHMCSampler as dhmc

### model
test.model.display_graph()
print(test.model.gen_prior_samples_code)
print(test.model.gen_pdf_code)