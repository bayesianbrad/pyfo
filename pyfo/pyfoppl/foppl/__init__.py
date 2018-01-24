#
# This file is part of PyFOPPL, an implementation of a First Order Probabilistic Programming Language in Python.
#
# License: MIT (see LICENSE.txt)
##
# 19. Dec 2017, Tobias Kohn
# 24. Jan 2018, Tobias Kohn
#
from . import test_distributions

class Options(object):
    """
    This class provides flags and general options to control the compilation process.

    `eager_conditionals`:
        Controls whether conditional statements (`if`-expressions) should be evaluated eagerly or lazily.
        _At the moment, only eager evaluation is supported._

    `uniform_conditionals`:
        If this flag is set to `True`, the compiler will transform all comparisons (except for equality) to be
        in the form `X >= 0`. For instance, `x < 5` will thus be transformed to `not (x-5 >= 0)`.

    `conditional_suffix`:
        A string suffix that is appended to conditional variables.

    `debug`:
        Print out additional information, e. g., about the nodes in the graph.

    `devectorize`
        When set to `True`, the compiler tries to unpack all vectors and lists and apply the operations on
        scalars only. When set to `False`, the compiler will leave vectors and try to avoid unpacking any
        of them.
    """

    eager_conditionals = True

    uniform_conditionals = True

    conditional_suffix = '.data[0]'

    debug =True

    devectorize = False

    log_model = None


# Stubs to make the Python-IDE happy

def sample(distr): return 0

def observe(distr, value): pass

def binomial(p): pass

def categorical(ps): pass

def normal(mu, sigma): pass

def interleave(a, b): return a