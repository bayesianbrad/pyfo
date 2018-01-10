import os
import sys
sys.path.insert(1, '~/Desktop/Pyfo/pyfo')
from pyfo.pyfoppl import foppl
import pyfo.distributions as dist
import tests.test_onegauss as test_onegauss
#
print("=" * 50)
print(test_onegauss.code)
print("=" * 50)
print(test_onegauss.graph)
print("=" * 50)
print(help(test_onegauss.model))
